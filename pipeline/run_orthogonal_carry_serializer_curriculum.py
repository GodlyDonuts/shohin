# if 0
"""
#endif
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif
#ifndef O_NOFOLLOW
#error "OCSC bootstrap requires O_NOFOLLOW"
#endif

#define OCSC_BOOTSTRAP_FD 20
#define OCSC_CHECKOUT_ROOT_FD 21
#define OCSC_INTERPRETER_FD 22
#define OCSC_RUNTIME_MANIFEST_FD 23
#define OCSC_SOURCE_FD_BASE 40
#define OCSC_RUNNER_FD (OCSC_SOURCE_FD_BASE + 3)
#define OCSC_RUNTIME_FD_BASE 64
#define OCSC_ATTESTATION_FD 30
#define OCSC_MAX_RUNTIME_RECORDS 1024
#define OCSC_MAX_HELD_RUNTIME_RECORDS 768
#define OCSC_RELOCATED_FD_BASE 2048
#define OCSC_MAX_FILE_BYTES (64U * 1024U * 1024U)
#define OCSC_MAX_MANIFEST_BYTES (8U * 1024U * 1024U)

static const char *const OCSC_SOURCE_PATHS[] = {
    "R12_ORTHOGONAL_CARRY_SERIALIZER_CURRICULUM_PREREG.md",
    "pipeline/generate_orthogonal_carry_serializer_curriculum.py",
    "pipeline/test_generate_orthogonal_carry_serializer_curriculum.py",
    "pipeline/run_orthogonal_carry_serializer_curriculum.py",
    "train/digitwise_protocol.py",
};

static const char *const OCSC_SOURCE_HASH_FLAGS[] = {
    "--prereg-sha256",
    "--generator-sha256",
    "--tests-sha256",
    "--runner-sha256",
    "--oracle-sha256",
};

static const char OCSC_PYTHON_LAUNCH_CODE[] =
    "import os,sys\n"
    "chunks=[]\n"
    "while True:\n"
    " b=os.read(30,1048576)\n"
    " if not b: break\n"
    " chunks.append(b)\n"
    "attestation=b''.join(chunks)\n"
    "argv=list(sys.argv[1:])\n"
    "positions=[i for i,v in enumerate(argv) if v=='--runner']\n"
    "if len(positions)!=1 or positions[0]+1>=len(argv): raise SystemExit('OCSC bootstrap rejected: runner argument mismatch')\n"
    "runner_path=argv[positions[0]+1]\n"
    "size=os.fstat(43).st_size\n"
    "payload=os.pread(43,size,0)\n"
    "if len(payload)!=size: raise SystemExit('OCSC bootstrap rejected: runner descriptor short read')\n"
    "namespace={'__builtins__':__builtins__,'__file__':runner_path,'__name__':'__main__','__package__':None,'_OCSC_EXTERNAL_BOOTSTRAP_ATTESTATION':attestation,'_OCSC_LOADED_RUNNER_PAYLOAD':payload}\n"
    "sys.argv=[runner_path]+argv\n"
    "exec(compile(payload,runner_path,'exec',dont_inherit=True,optimize=0),namespace)\n";

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_length;
} sha256_context;

typedef struct {
    char role[24];
    char sha256[65];
    char path[PATH_MAX];
    int descriptor;
    struct stat metadata;
} runtime_record;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} byte_buffer;

static uint32_t rotate_right(uint32_t value, unsigned int count) {
    return (value >> count) | (value << (32U - count));
}

static void sha256_transform(sha256_context *context, const unsigned char block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    unsigned int index;

    for (index = 0; index < 16; ++index) {
        size_t offset = (size_t)index * 4U;
        words[index] = ((uint32_t)block[offset] << 24U) |
                       ((uint32_t)block[offset + 1U] << 16U) |
                       ((uint32_t)block[offset + 2U] << 8U) |
                       (uint32_t)block[offset + 3U];
    }
    for (index = 16; index < 64; ++index) {
        uint32_t first = rotate_right(words[index - 15U], 7U) ^
                         rotate_right(words[index - 15U], 18U) ^
                         (words[index - 15U] >> 3U);
        uint32_t second = rotate_right(words[index - 2U], 17U) ^
                          rotate_right(words[index - 2U], 19U) ^
                          (words[index - 2U] >> 10U);
        words[index] = words[index - 16U] + first + words[index - 7U] + second;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (index = 0; index < 64; ++index) {
        uint32_t sum_one = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                           rotate_right(e, 25U);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temporary_one = h + sum_one + choose + constants[index] + words[index];
        uint32_t sum_zero = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                            rotate_right(a, 22U);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary_two = sum_zero + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary_one;
        d = c;
        c = b;
        b = a;
        a = temporary_one + temporary_two;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_initialize(sha256_context *context) {
    static const uint32_t initial[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    memcpy(context->state, initial, sizeof(initial));
    context->bit_count = 0U;
    context->block_length = 0U;
}

static void sha256_update(sha256_context *context, const unsigned char *data, size_t length) {
    size_t offset = 0U;
    context->bit_count += (uint64_t)length * 8U;
    while (offset < length) {
        size_t available = 64U - context->block_length;
        size_t take = length - offset < available ? length - offset : available;
        memcpy(context->block + context->block_length, data + offset, take);
        context->block_length += take;
        offset += take;
        if (context->block_length == 64U) {
            sha256_transform(context, context->block);
            context->block_length = 0U;
        }
    }
}

static void sha256_finalize(sha256_context *context, unsigned char digest[32]) {
    uint64_t bit_count = context->bit_count;
    unsigned int index;
    context->block[context->block_length++] = 0x80U;
    if (context->block_length > 56U) {
        while (context->block_length < 64U) {
            context->block[context->block_length++] = 0U;
        }
        sha256_transform(context, context->block);
        context->block_length = 0U;
    }
    while (context->block_length < 56U) {
        context->block[context->block_length++] = 0U;
    }
    for (index = 0; index < 8U; ++index) {
        context->block[63U - index] = (unsigned char)(bit_count >> (index * 8U));
    }
    sha256_transform(context, context->block);
    for (index = 0; index < 8U; ++index) {
        digest[index * 4U] = (unsigned char)(context->state[index] >> 24U);
        digest[index * 4U + 1U] = (unsigned char)(context->state[index] >> 16U);
        digest[index * 4U + 2U] = (unsigned char)(context->state[index] >> 8U);
        digest[index * 4U + 3U] = (unsigned char)context->state[index];
    }
}

static void digest_hex(const unsigned char digest[32], char output[65]) {
    static const char alphabet[] = "0123456789abcdef";
    size_t index;
    for (index = 0; index < 32U; ++index) {
        output[index * 2U] = alphabet[digest[index] >> 4U];
        output[index * 2U + 1U] = alphabet[digest[index] & 0x0fU];
    }
    output[64] = '\0';
}

static void reject(const char *format, ...) {
    va_list arguments;
    fputs("OCSC external bootstrap rejected: ", stderr);
    va_start(arguments, format);
    vfprintf(stderr, format, arguments);
    va_end(arguments);
    fputc('\n', stderr);
    exit(2);
}

static void require_ascii_field(const char *value, const char *label) {
    const unsigned char *cursor = (const unsigned char *)value;
    if (value == NULL || value[0] == '\0') {
        reject("%s is empty", label);
    }
    while (*cursor != 0U) {
        if (*cursor < 0x20U || *cursor > 0x7eU || *cursor == '\t') {
            reject("%s is not strict printable ASCII", label);
        }
        ++cursor;
    }
}

static void require_hash(const char *value, const char *label) {
    size_t index;
    if (value == NULL || strlen(value) != 64U) {
        reject("%s is not lowercase SHA-256", label);
    }
    for (index = 0; index < 64U; ++index) {
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f'))) {
            reject("%s is not lowercase SHA-256", label);
        }
    }
}

static const char *one_argument(int argc, char **argv, const char *flag) {
    const char *result = NULL;
    int index;
    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], flag) == 0) {
            if (result != NULL || index + 1 >= argc) {
                reject("missing or duplicate %s", flag);
            }
            result = argv[index + 1];
        }
    }
    if (result == NULL) {
        reject("missing or duplicate %s", flag);
    }
    require_ascii_field(result, flag);
    return result;
}

static void require_lexical_absolute_path(const char *path, const char *label) {
    const char *cursor;
    const char *component;
    if (path == NULL || path[0] != '/' || path[1] == '\0') {
        reject("%s must be one non-root absolute path", label);
    }
    require_ascii_field(path, label);
    cursor = path + 1;
    while (*cursor != '\0') {
        size_t length;
        component = cursor;
        while (*cursor != '/' && *cursor != '\0') {
            ++cursor;
        }
        length = (size_t)(cursor - component);
        if (length == 0U || (length == 1U && component[0] == '.') ||
            (length == 2U && component[0] == '.' && component[1] == '.')) {
            reject("%s contains an empty or dot component", label);
        }
        if (*cursor == '/') {
            ++cursor;
            if (*cursor == '\0') {
                reject("%s has a trailing slash", label);
            }
        }
    }
}

static int open_absolute_nofollow(const char *path, int final_flags, const char *label) {
    int current;
    const char *cursor;
    require_lexical_absolute_path(path, label);
    current = open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (current < 0) {
        reject("cannot pin filesystem root for %s: %s", label, strerror(errno));
    }
    cursor = path + 1;
    while (*cursor != '\0') {
        const char *start = cursor;
        char component[NAME_MAX + 1];
        size_t length;
        int next;
        int flags;
        struct stat held;
        struct stat entry;
        while (*cursor != '/' && *cursor != '\0') {
            ++cursor;
        }
        length = (size_t)(cursor - start);
        if (length == 0U || length > NAME_MAX) {
            close(current);
            reject("%s has an invalid component length", label);
        }
        memcpy(component, start, length);
        component[length] = '\0';
        flags = *cursor == '\0'
                    ? final_flags | O_NOFOLLOW | O_CLOEXEC
                    : O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
        next = openat(current, component, flags);
        if (next < 0) {
            close(current);
            reject("%s component is not safely openable: %s", label, strerror(errno));
        }
        if (fstat(next, &held) != 0 || fstatat(current, component, &entry, AT_SYMLINK_NOFOLLOW) != 0 ||
            held.st_dev != entry.st_dev || held.st_ino != entry.st_ino || held.st_mode != entry.st_mode) {
            close(next);
            close(current);
            reject("%s component identity changed during traversal", label);
        }
        close(current);
        current = next;
        if (*cursor == '/') {
            ++cursor;
        }
    }
    return current;
}

static int open_relative_nofollow(int root_fd, const char *relative, int final_flags, const char *label) {
    int current;
    const char *cursor;
    if (relative == NULL || relative[0] == '\0' || relative[0] == '/') {
        reject("%s must be a non-empty relative path", label);
    }
    require_ascii_field(relative, label);
    current = dup(root_fd);
    if (current < 0) {
        reject("cannot duplicate checkout root for %s", label);
    }
    if (fcntl(current, F_SETFD, FD_CLOEXEC) != 0) {
        close(current);
        reject("cannot protect checkout root duplicate for %s", label);
    }
    cursor = relative;
    while (*cursor != '\0') {
        const char *start = cursor;
        char component[NAME_MAX + 1];
        size_t length;
        int next;
        int flags;
        struct stat held;
        struct stat entry;
        while (*cursor != '/' && *cursor != '\0') {
            ++cursor;
        }
        length = (size_t)(cursor - start);
        if (length == 0U || length > NAME_MAX ||
            (length == 1U && start[0] == '.') ||
            (length == 2U && start[0] == '.' && start[1] == '.')) {
            close(current);
            reject("%s contains an empty or dot component", label);
        }
        memcpy(component, start, length);
        component[length] = '\0';
        flags = *cursor == '\0'
                    ? final_flags | O_NOFOLLOW | O_CLOEXEC
                    : O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
        next = openat(current, component, flags);
        if (next < 0) {
            close(current);
            reject("%s component is not safely openable: %s", label, strerror(errno));
        }
        if (fstat(next, &held) != 0 || fstatat(current, component, &entry, AT_SYMLINK_NOFOLLOW) != 0 ||
            held.st_dev != entry.st_dev || held.st_ino != entry.st_ino || held.st_mode != entry.st_mode) {
            close(next);
            close(current);
            reject("%s component identity changed during traversal", label);
        }
        close(current);
        current = next;
        if (*cursor == '/') {
            ++cursor;
            if (*cursor == '\0') {
                close(current);
                reject("%s has a trailing slash", label);
            }
        }
    }
    return current;
}

static void validate_regular_file(int descriptor, const char *label, struct stat *metadata) {
    if (fstat(descriptor, metadata) != 0) {
        reject("cannot stat %s: %s", label, strerror(errno));
    }
    if (!S_ISREG(metadata->st_mode) || metadata->st_nlink != 1 ||
        (metadata->st_mode & 0002) != 0 || metadata->st_size < 0 ||
        (uint64_t)metadata->st_size > OCSC_MAX_FILE_BYTES) {
        reject("%s is not one immutable regular file", label);
    }
}

static void hash_descriptor(int descriptor, const struct stat *expected, char output[65], const char *label) {
    sha256_context context;
    unsigned char digest[32];
    unsigned char block[1024U * 1024U];
    off_t offset = 0;
    struct stat before;
    struct stat after;
    ssize_t read_count;
    sha256_initialize(&context);
    if (fstat(descriptor, &before) != 0) {
        reject("cannot stat %s before hashing", label);
    }
    while (offset < before.st_size) {
        size_t request = (uint64_t)(before.st_size - offset) < sizeof(block)
                             ? (size_t)(before.st_size - offset)
                             : sizeof(block);
        read_count = pread(descriptor, block, request, offset);
        if (read_count <= 0) {
            reject("cannot completely hash %s", label);
        }
        sha256_update(&context, block, (size_t)read_count);
        offset += read_count;
    }
    if (fstat(descriptor, &after) != 0 || offset != before.st_size ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_mode != after.st_mode || before.st_nlink != after.st_nlink ||
        before.st_size != after.st_size || before.st_mtime != after.st_mtime ||
        before.st_ctime != after.st_ctime) {
        reject("%s changed while it was hashed", label);
    }
    if (expected != NULL && (expected->st_dev != before.st_dev || expected->st_ino != before.st_ino ||
                             expected->st_mode != before.st_mode || expected->st_nlink != before.st_nlink ||
                             expected->st_size != before.st_size)) {
        reject("%s descriptor identity mismatch", label);
    }
    sha256_finalize(&context, digest);
    digest_hex(digest, output);
}

static unsigned char *read_descriptor_bytes(int descriptor, size_t maximum, size_t *length, const char *label) {
    struct stat metadata;
    unsigned char *payload;
    size_t offset = 0U;
    if (fstat(descriptor, &metadata) != 0 || metadata.st_size <= 0 ||
        (uint64_t)metadata.st_size > maximum) {
        reject("%s has an invalid byte length", label);
    }
    payload = (unsigned char *)malloc((size_t)metadata.st_size + 1U);
    if (payload == NULL) {
        reject("cannot allocate %s bytes", label);
    }
    while (offset < (size_t)metadata.st_size) {
        ssize_t count = pread(descriptor, payload + offset, (size_t)metadata.st_size - offset, (off_t)offset);
        if (count <= 0) {
            free(payload);
            reject("cannot read all %s bytes", label);
        }
        offset += (size_t)count;
    }
    payload[offset] = 0U;
    *length = offset;
    return payload;
}

static void initialize_buffer(byte_buffer *buffer) {
    buffer->capacity = 32768U;
    buffer->length = 0U;
    buffer->data = (char *)malloc(buffer->capacity);
    if (buffer->data == NULL) {
        reject("cannot allocate attestation buffer");
    }
}

static void append_buffer(byte_buffer *buffer, const char *format, ...) {
    va_list arguments;
    va_list copied;
    int required;
    while (1) {
        size_t available = buffer->capacity - buffer->length;
        va_start(arguments, format);
        va_copy(copied, arguments);
        required = vsnprintf(buffer->data + buffer->length, available, format, copied);
        va_end(copied);
        va_end(arguments);
        if (required < 0) {
            reject("cannot encode bootstrap attestation");
        }
        if ((size_t)required < available) {
            buffer->length += (size_t)required;
            return;
        }
        while (buffer->capacity - buffer->length <= (size_t)required) {
            buffer->capacity *= 2U;
        }
        buffer->data = (char *)realloc(buffer->data, buffer->capacity);
        if (buffer->data == NULL) {
            reject("cannot grow bootstrap attestation");
        }
    }
}

static void make_inheritable(int descriptor, int target, const char *label) {
    if (descriptor != target) {
        if (dup2(descriptor, target) < 0) {
            reject("cannot assign inherited descriptor for %s", label);
        }
        close(descriptor);
    }
    if (fcntl(target, F_SETFD, 0) != 0) {
        reject("cannot preserve inherited descriptor for %s", label);
    }
}

static int relocate_descriptor(int descriptor, const char *label) {
    int relocated = fcntl(descriptor, F_DUPFD_CLOEXEC, OCSC_RELOCATED_FD_BASE);
    if (relocated < 0) {
        reject("cannot relocate retained descriptor for %s", label);
    }
    close(descriptor);
    return relocated;
}

static int parse_runtime_manifest(
    unsigned char *payload,
    size_t length,
    const char *bootstrap_path,
    const char *bootstrap_sha256,
    const char *python_path,
    const char *python_sha256,
    runtime_record records[OCSC_MAX_RUNTIME_RECORDS],
    size_t *record_count
) {
    const char *schema = "shohin-ocsc-external-runtime-closure-v1\n";
    char *cursor;
    char *end;
    char previous_line[PATH_MAX + 128];
    int has_previous_line = 0;
    size_t count = 0U;
    size_t held_count = 0U;
    int saw_bootstrap = 0;
    int saw_python = 0;
    if (length <= strlen(schema) || memcmp(payload, schema, strlen(schema)) != 0 ||
        payload[length - 1U] != '\n' || memchr(payload, '\r', length) != NULL ||
        memchr(payload, 0, length) != NULL) {
        reject("runtime manifest framing mismatch");
    }
    cursor = (char *)payload + strlen(schema);
    end = (char *)payload + length;
    while (cursor < end) {
        char *line_end = memchr(cursor, '\n', (size_t)(end - cursor));
        char *first_tab;
        char *second_tab;
        runtime_record *record;
        char actual_hash[65];
        if (line_end == NULL || line_end == cursor || count >= OCSC_MAX_RUNTIME_RECORDS) {
            reject("runtime manifest line inventory mismatch");
        }
        *line_end = '\0';
        if (strlen(cursor) >= sizeof(previous_line)) {
            reject("runtime manifest line is too long");
        }
        if (has_previous_line && strcmp(previous_line, cursor) >= 0) {
            reject("runtime manifest lines are not unique and sorted");
        }
        strcpy(previous_line, cursor);
        has_previous_line = 1;
        first_tab = strchr(cursor, '\t');
        second_tab = first_tab == NULL ? NULL : strchr(first_tab + 1, '\t');
        if (first_tab == NULL || second_tab == NULL || strchr(second_tab + 1, '\t') != NULL) {
            reject("runtime manifest record framing mismatch");
        }
        *first_tab = '\0';
        *second_tab = '\0';
        record = &records[count];
        if (strlen(cursor) >= sizeof(record->role) || strlen(first_tab + 1) != 64U ||
            strlen(second_tab + 1) >= sizeof(record->path)) {
            reject("runtime manifest record length mismatch");
        }
        strcpy(record->role, cursor);
        strcpy(record->sha256, first_tab + 1);
        strcpy(record->path, second_tab + 1);
        require_hash(record->sha256, "runtime manifest record hash");
        require_lexical_absolute_path(record->path, "runtime manifest record path");
        if (strcmp(record->role, "bootstrap") != 0 && strcmp(record->role, "python") != 0 &&
            strcmp(record->role, "runtime-held") != 0 && strcmp(record->role, "runtime-inventory") != 0) {
            reject("runtime manifest role mismatch");
        }
        {
            size_t earlier;
            for (earlier = 0U; earlier < count; ++earlier) {
                if (strcmp(records[earlier].path, record->path) == 0) {
                    reject("runtime manifest path is duplicated");
                }
            }
        }
        record->descriptor = open_absolute_nofollow(record->path, O_RDONLY, "runtime manifest file");
        validate_regular_file(record->descriptor, "runtime manifest file", &record->metadata);
        hash_descriptor(record->descriptor, &record->metadata, actual_hash, "runtime manifest file");
        if (strcmp(actual_hash, record->sha256) != 0) {
            reject("runtime manifest file does not match approved bytes");
        }
        if (strcmp(record->role, "bootstrap") == 0) {
            if (saw_bootstrap || strcmp(record->path, bootstrap_path) != 0 ||
                strcmp(record->sha256, bootstrap_sha256) != 0) {
                reject("runtime manifest bootstrap binding mismatch");
            }
            saw_bootstrap = 1;
        } else if (strcmp(record->role, "python") == 0) {
            if (saw_python || strcmp(record->path, python_path) != 0 ||
                strcmp(record->sha256, python_sha256) != 0) {
                reject("runtime manifest Python binding mismatch");
            }
            saw_python = 1;
        } else if (strcmp(record->role, "runtime-held") == 0) {
            if (held_count >= OCSC_MAX_HELD_RUNTIME_RECORDS) {
                reject("runtime manifest exceeds retained-descriptor limit");
            }
            ++held_count;
        }
        cursor = line_end + 1;
        ++count;
    }
    if (!saw_bootstrap || !saw_python || held_count == 0U || count < 3U) {
        reject("runtime manifest closure is incomplete");
    }
    *record_count = count;
    return (int)held_count;
}

static void write_all(int descriptor, const char *payload, size_t length) {
    size_t offset = 0U;
    while (offset < length) {
        ssize_t count = write(descriptor, payload + offset, length - offset);
        if (count <= 0) {
            reject("cannot write bootstrap attestation");
        }
        offset += (size_t)count;
    }
}

int main(int argc, char **argv) {
    const char *bootstrap_sha256 = one_argument(argc, argv, "--bootstrap-sha256");
    const char *runtime_manifest_path = one_argument(argc, argv, "--runtime-manifest");
    const char *runtime_manifest_sha256 = one_argument(argc, argv, "--runtime-manifest-sha256");
    const char *python_path = one_argument(argc, argv, "--python");
    const char *python_sha256 = one_argument(argc, argv, "--python-sha256");
    const char *checkout_root_path = one_argument(argc, argv, "--checkout-root");
    const char *runner_path = one_argument(argc, argv, "--runner");
    int bootstrap_fd;
    int checkout_root_fd;
    int python_fd;
    int manifest_fd;
    int source_fds[5];
    runtime_record runtime_records[OCSC_MAX_RUNTIME_RECORDS];
    size_t runtime_record_count = 0U;
    unsigned char *manifest_payload;
    size_t manifest_length;
    struct stat bootstrap_metadata;
    struct stat checkout_metadata;
    struct stat python_metadata;
    struct stat manifest_metadata;
    struct stat source_metadata[5];
    char bootstrap_actual[65];
    char python_actual[65];
    char manifest_actual[65];
    char source_actual[5][65];
    char source_absolute[5][PATH_MAX];
    byte_buffer attestation;
    sha256_context attestation_hash_context;
    unsigned char attestation_digest[32];
    char attestation_hash[65];
    int attestation_fd;
    char attestation_template[] = "/tmp/shohin-ocsc-attestation-XXXXXX";
    size_t index;
    size_t held_index = 0U;
    char **python_argv;
    char *clean_environment[] = {
        "LC_ALL=C",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        NULL,
    };

    require_hash(bootstrap_sha256, "bootstrap hash");
    require_hash(runtime_manifest_sha256, "runtime manifest hash");
    require_hash(python_sha256, "Python hash");
    require_lexical_absolute_path(argv[0], "bootstrap executable path");
    require_lexical_absolute_path(runtime_manifest_path, "runtime manifest path");
    require_lexical_absolute_path(python_path, "Python path");
    require_lexical_absolute_path(checkout_root_path, "checkout root path");
    require_lexical_absolute_path(runner_path, "runner path");

    bootstrap_fd = open_absolute_nofollow(argv[0], O_RDONLY, "bootstrap executable");
    validate_regular_file(bootstrap_fd, "bootstrap executable", &bootstrap_metadata);
    hash_descriptor(bootstrap_fd, &bootstrap_metadata, bootstrap_actual, "bootstrap executable");
    if (strcmp(bootstrap_actual, bootstrap_sha256) != 0) {
        reject("bootstrap executable bytes do not match approved hash");
    }
    checkout_root_fd = open_absolute_nofollow(
        checkout_root_path, O_RDONLY | O_DIRECTORY, "checkout root");
    if (fstat(checkout_root_fd, &checkout_metadata) != 0 || !S_ISDIR(checkout_metadata.st_mode)) {
        reject("checkout root identity mismatch");
    }
    python_fd = open_absolute_nofollow(python_path, O_RDONLY, "Python interpreter");
    validate_regular_file(python_fd, "Python interpreter", &python_metadata);
    hash_descriptor(python_fd, &python_metadata, python_actual, "Python interpreter");
    if (strcmp(python_actual, python_sha256) != 0) {
        reject("Python bytes do not match approved hash");
    }
    manifest_fd = open_absolute_nofollow(runtime_manifest_path, O_RDONLY, "runtime manifest");
    validate_regular_file(manifest_fd, "runtime manifest", &manifest_metadata);
    if ((uint64_t)manifest_metadata.st_size > OCSC_MAX_MANIFEST_BYTES) {
        reject("runtime manifest is too large");
    }
    hash_descriptor(manifest_fd, &manifest_metadata, manifest_actual, "runtime manifest");
    if (strcmp(manifest_actual, runtime_manifest_sha256) != 0) {
        reject("runtime manifest bytes do not match approved hash");
    }
    manifest_payload = read_descriptor_bytes(
        manifest_fd, OCSC_MAX_MANIFEST_BYTES, &manifest_length, "runtime manifest");
    parse_runtime_manifest(
        manifest_payload,
        manifest_length,
        argv[0],
        bootstrap_actual,
        python_path,
        python_actual,
        runtime_records,
        &runtime_record_count);
    free(manifest_payload);

    for (index = 0U; index < 5U; ++index) {
        const char *expected_hash = one_argument(argc, argv, OCSC_SOURCE_HASH_FLAGS[index]);
        int count;
        require_hash(expected_hash, OCSC_SOURCE_PATHS[index]);
        count = snprintf(
            source_absolute[index],
            sizeof(source_absolute[index]),
            "%s/%s",
            checkout_root_path,
            OCSC_SOURCE_PATHS[index]);
        if (count < 0 || (size_t)count >= sizeof(source_absolute[index])) {
            reject("source absolute path is too long");
        }
        source_fds[index] = open_relative_nofollow(
            checkout_root_fd, OCSC_SOURCE_PATHS[index], O_RDONLY, OCSC_SOURCE_PATHS[index]);
        validate_regular_file(source_fds[index], OCSC_SOURCE_PATHS[index], &source_metadata[index]);
        hash_descriptor(source_fds[index], &source_metadata[index], source_actual[index], OCSC_SOURCE_PATHS[index]);
        if (strcmp(source_actual[index], expected_hash) != 0) {
            reject("%s does not match its approved hash", OCSC_SOURCE_PATHS[index]);
        }
    }
    if (strcmp(runner_path, source_absolute[3]) != 0) {
        reject("runner path is not the reviewed checkout-relative path");
    }

    initialize_buffer(&attestation);
    append_buffer(&attestation, "version\tshohin-ocsc-external-bootstrap-attestation-v1\n");
    append_buffer(
        &attestation,
        "bootstrap\t%s\t%s\t%lld\t%llu\t%llu\t%d\n",
        argv[0],
        bootstrap_actual,
        (long long)bootstrap_metadata.st_size,
        (unsigned long long)bootstrap_metadata.st_dev,
        (unsigned long long)bootstrap_metadata.st_ino,
        OCSC_BOOTSTRAP_FD);
    append_buffer(
        &attestation,
        "checkout-root\t%s\t-\t0\t%llu\t%llu\t%d\n",
        checkout_root_path,
        (unsigned long long)checkout_metadata.st_dev,
        (unsigned long long)checkout_metadata.st_ino,
        OCSC_CHECKOUT_ROOT_FD);
    append_buffer(
        &attestation,
        "interpreter\t%s\t%s\t%lld\t%llu\t%llu\t%d\n",
        python_path,
        python_actual,
        (long long)python_metadata.st_size,
        (unsigned long long)python_metadata.st_dev,
        (unsigned long long)python_metadata.st_ino,
        OCSC_INTERPRETER_FD);
    append_buffer(
        &attestation,
        "runtime-manifest\t%s\t%s\t%lld\t%llu\t%llu\t%d\n",
        runtime_manifest_path,
        manifest_actual,
        (long long)manifest_metadata.st_size,
        (unsigned long long)manifest_metadata.st_dev,
        (unsigned long long)manifest_metadata.st_ino,
        OCSC_RUNTIME_MANIFEST_FD);
    for (index = 0U; index < 5U; ++index) {
        append_buffer(
            &attestation,
            "source\t%s\t%s\t%s\t%lld\t%llu\t%llu\t%d\n",
            OCSC_SOURCE_PATHS[index],
            source_absolute[index],
            source_actual[index],
            (long long)source_metadata[index].st_size,
            (unsigned long long)source_metadata[index].st_dev,
            (unsigned long long)source_metadata[index].st_ino,
            OCSC_SOURCE_FD_BASE + (int)index);
    }
    for (index = 0U; index < runtime_record_count; ++index) {
        runtime_record *record = &runtime_records[index];
        int inherited_fd = -1;
        if (strcmp(record->role, "runtime-held") == 0) {
            inherited_fd = OCSC_RUNTIME_FD_BASE + (int)held_index;
            ++held_index;
        }
        append_buffer(
            &attestation,
            "runtime\t%s\t%s\t%s\t%lld\t%llu\t%llu\t%d\n",
            record->role,
            record->path,
            record->sha256,
            (long long)record->metadata.st_size,
            (unsigned long long)record->metadata.st_dev,
            (unsigned long long)record->metadata.st_ino,
            inherited_fd);
    }
    sha256_initialize(&attestation_hash_context);
    sha256_update(
        &attestation_hash_context,
        (const unsigned char *)attestation.data,
        attestation.length);
    sha256_finalize(&attestation_hash_context, attestation_digest);
    digest_hex(attestation_digest, attestation_hash);
    append_buffer(&attestation, "attestation-sha256\t%s\n", attestation_hash);

    bootstrap_fd = relocate_descriptor(bootstrap_fd, "bootstrap executable");
    checkout_root_fd = relocate_descriptor(checkout_root_fd, "checkout root");
    python_fd = relocate_descriptor(python_fd, "Python interpreter");
    manifest_fd = relocate_descriptor(manifest_fd, "runtime manifest");
    for (index = 0U; index < 5U; ++index) {
        source_fds[index] = relocate_descriptor(source_fds[index], OCSC_SOURCE_PATHS[index]);
    }
    for (index = 0U; index < runtime_record_count; ++index) {
        runtime_record *record = &runtime_records[index];
        if (strcmp(record->role, "runtime-held") == 0) {
            record->descriptor = relocate_descriptor(record->descriptor, "held runtime file");
        } else {
            close(record->descriptor);
            record->descriptor = -1;
        }
    }

    make_inheritable(bootstrap_fd, OCSC_BOOTSTRAP_FD, "bootstrap executable");
    make_inheritable(checkout_root_fd, OCSC_CHECKOUT_ROOT_FD, "checkout root");
    make_inheritable(python_fd, OCSC_INTERPRETER_FD, "Python interpreter");
    make_inheritable(manifest_fd, OCSC_RUNTIME_MANIFEST_FD, "runtime manifest");
    for (index = 0U; index < 5U; ++index) {
        make_inheritable(source_fds[index], OCSC_SOURCE_FD_BASE + (int)index, OCSC_SOURCE_PATHS[index]);
    }
    held_index = 0U;
    for (index = 0U; index < runtime_record_count; ++index) {
        runtime_record *record = &runtime_records[index];
        if (strcmp(record->role, "runtime-held") == 0) {
            make_inheritable(
                record->descriptor,
                OCSC_RUNTIME_FD_BASE + (int)held_index,
                "held runtime file");
            ++held_index;
        }
    }
    attestation_fd = mkstemp(attestation_template);
    if (attestation_fd < 0 || unlink(attestation_template) != 0) {
        reject("cannot create anonymous bootstrap attestation descriptor");
    }
    write_all(attestation_fd, attestation.data, attestation.length);
    if (fsync(attestation_fd) != 0 || lseek(attestation_fd, 0, SEEK_SET) != 0) {
        reject("cannot make bootstrap attestation descriptor durable and readable");
    }
    free(attestation.data);
    make_inheritable(attestation_fd, OCSC_ATTESTATION_FD, "bootstrap attestation");

    python_argv = (char **)calloc((size_t)argc + 6U, sizeof(char *));
    if (python_argv == NULL) {
        reject("cannot allocate Python argv");
    }
    python_argv[0] = (char *)python_path;
    python_argv[1] = "-I";
    python_argv[2] = "-S";
    python_argv[3] = "-B";
    python_argv[4] = "-c";
    python_argv[5] = (char *)OCSC_PYTHON_LAUNCH_CODE;
    for (index = 1U; index < (size_t)argc; ++index) {
        python_argv[index + 5U] = argv[index];
    }
    python_argv[(size_t)argc + 5U] = NULL;

#if defined(__linux__)
    fexecve(OCSC_INTERPRETER_FD, python_argv, clean_environment);
#else
    execve(python_path, python_argv, clean_environment);
#endif
    reject("cannot execute pinned Python interpreter: %s", strerror(errno));
    return 2;
}

#if 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import sysconfig
import types


SOURCE_PATHS = (
    "R12_ORTHOGONAL_CARRY_SERIALIZER_CURRICULUM_PREREG.md",
    "pipeline/generate_orthogonal_carry_serializer_curriculum.py",
    "pipeline/test_generate_orthogonal_carry_serializer_curriculum.py",
    "pipeline/run_orthogonal_carry_serializer_curriculum.py",
    "train/digitwise_protocol.py",
)
RUNNER_RELATIVE_PATH = "pipeline/run_orthogonal_carry_serializer_curriculum.py"
GENERATOR_RELATIVE_PATH = "pipeline/generate_orthogonal_carry_serializer_curriculum.py"
HASH_ARGUMENTS = {
    SOURCE_PATHS[0]: "prereg_sha256",
    SOURCE_PATHS[1]: "generator_sha256",
    SOURCE_PATHS[2]: "tests_sha256",
    SOURCE_PATHS[3]: "runner_sha256",
    SOURCE_PATHS[4]: "oracle_sha256",
}
QUALIFICATION_CHECKS = (
    "external_bootstrap_source_bound",
    "all_consumed_source_bytes_pinned",
    "runtime_closure_complete_before_action",
    "real_tokenizer_registry_consumed",
    "cross_node_distinct_hosts",
    "same_lustre_mount_and_output_inode",
    "production_broker_transfer_complete",
    "publication_path_complete",
    "renameat2_noreplace_real",
    "descriptor_relative_io",
    "file_fsync_after_chmod",
    "stage_and_parent_fsync",
    "kernel_flock_live_exclusion",
    "stale_lease_observed_without_mutation",
    "all_crash_evidence_permanently_retained",
    "canonical_death_recovery",
    "collision_rejected",
    "path_substitution_rejected",
    "coherent_forgery_foreign_child_preserved",
    "partial_child_preserved",
    "foreign_replacement_preserved",
    "runtime_shadow_import_rejected",
    "injected_io_fail_closed",
    "strict_full_readback",
    "report_marker_receipt_event_derived",
    "permanent_evidence_inventory_recorded",
)
QUALIFICATION_CRASH_POINTS = (
    "stage-created-before-journal",
    "journal-durable-before-first-artifact",
    "partial-artifact-write",
    "stage-fsync-before-rename",
    "canonical-before-parent-fsync",
)

_EXTERNAL_ATTESTATION_BYTES = globals().get("_OCSC_EXTERNAL_BOOTSTRAP_ATTESTATION")
_LOADED_RUNNER_PAYLOAD = globals().get("_OCSC_LOADED_RUNNER_PAYLOAD")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_hash(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(label + " must be one lowercase SHA-256")
    return value


def lexical_absolute_path(path: Path | str, label: str) -> str:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.startswith("/") or raw == "/":
        raise RuntimeError(label + " must be one non-root absolute path")
    try:
        raw.encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeError(label + " must be ASCII") from error
    if raw.endswith("/") or any(
        component in {"", ".", ".."} for component in raw.split("/")[1:]
    ):
        raise RuntimeError(label + " contains an empty or dot component")
    return raw


def relative_components(relative: str, label: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        raise RuntimeError(label + " must be one relative path")
    try:
        relative.encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeError(label + " must be ASCII") from error
    components = tuple(relative.split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise RuntimeError(label + " contains an empty or dot component")
    return components


def descriptor_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def open_absolute_nofollow(
    path: Path | str,
    label: str,
    *,
    directory: bool = False,
) -> int:
    absolute = lexical_absolute_path(path, label)
    descriptor = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        components = absolute.split("/")[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            if not final or directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                held = os.fstat(next_descriptor)
                entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if descriptor_state(held) != descriptor_state(entry):
                    raise RuntimeError(label + " component identity changed")
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_relative_nofollow(
    root_fd: int,
    relative: str,
    label: str,
    *,
    directory: bool = False,
) -> int:
    components = relative_components(relative, label)
    descriptor = os.dup(root_fd)
    try:
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            if not final or directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                held = os.fstat(next_descriptor)
                entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if descriptor_state(held) != descriptor_state(entry):
                    raise RuntimeError(label + " component identity changed")
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_descriptor(descriptor: int, label: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError(label + " must be one regular file with one hard link")
    offset = 0
    chunks = []
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    payload = b"".join(chunks)
    if len(payload) != before.st_size or descriptor_state(before) != descriptor_state(
        after
    ):
        raise RuntimeError(label + " changed while its descriptor was read")
    return payload, before


def file_contract(path: str, payload: bytes, metadata: os.stat_result) -> dict:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "hard_links": metadata.st_nlink,
    }


def open_pinned_file(path: Path, label: str) -> tuple[int, bytes, dict]:
    descriptor = open_absolute_nofollow(path, label)
    try:
        payload, metadata = read_descriptor(descriptor, label)
        contract = file_contract(lexical_absolute_path(path, label), payload, metadata)
        verify_pinned_path(descriptor, payload, contract, label)
        return descriptor, payload, contract
    except BaseException:
        os.close(descriptor)
        raise


def verify_pinned_path(
    descriptor: int,
    payload: bytes,
    contract: dict,
    label: str,
) -> None:
    reread, metadata = read_descriptor(descriptor, label)
    live_fd = open_absolute_nofollow(contract["path"], label)
    try:
        live_payload, live_metadata = read_descriptor(live_fd, label)
    finally:
        os.close(live_fd)
    if (
        reread != payload
        or live_payload != payload
        or descriptor_state(metadata) != descriptor_state(live_metadata)
        or file_contract(contract["path"], reread, metadata) != contract
    ):
        raise RuntimeError(label + " changed after it was pinned")


def verify_executed_python_identity(
    descriptor: int,
    *,
    platform_name: str,
    execution_path: Path,
) -> dict:
    """Bind a Linux execution path to the retained interpreter inode."""

    if platform_name != "linux":
        return {
            "platform": platform_name,
            "authoritative_linux_execution_identity": False,
        }
    payload, held = read_descriptor(descriptor, "executed Python interpreter")
    path = lexical_absolute_path(execution_path, "executed Python path")
    live_fd = open_absolute_nofollow(path, "executed Python path")
    try:
        live_payload, live = read_descriptor(live_fd, "executed Python path")
    finally:
        os.close(live_fd)
    if live_payload != payload or descriptor_state(live) != descriptor_state(held):
        raise RuntimeError(
            "execution path does not name the interpreter executing the runner"
        )
    return {
        "platform": platform_name,
        "authoritative_linux_execution_identity": True,
        "interpreter": file_contract(path, payload, held),
    }


def _parse_integer(value: str, label: str) -> int:
    if not value or any(character not in "0123456789" for character in value):
        raise RuntimeError(label + " is not a nonnegative decimal integer")
    return int(value)


def parse_external_attestation(payload: bytes) -> dict:
    if (
        not isinstance(payload, bytes)
        or not payload.endswith(b"\n")
        or b"\r" in payload
    ):
        raise RuntimeError("external bootstrap attestation framing mismatch")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("external bootstrap attestation must be ASCII") from error
    if len(lines) < 11 or lines[0] != (
        "version\tshohin-ocsc-external-bootstrap-attestation-v1"
    ):
        raise RuntimeError("external bootstrap attestation schema mismatch")
    final = lines[-1].split("\t")
    if len(final) != 2 or final[0] != "attestation-sha256":
        raise RuntimeError("external bootstrap attestation hash record mismatch")
    require_hash(final[1], "external bootstrap attestation hash")
    prefix = ("\n".join(lines[:-1]) + "\n").encode("ascii")
    if hashlib.sha256(prefix).hexdigest() != final[1]:
        raise RuntimeError("external bootstrap attestation hash mismatch")
    result = {
        "attestation_sha256": final[1],
        "bootstrap": None,
        "checkout_root": None,
        "interpreter": None,
        "runtime_manifest": None,
        "sources": {},
        "runtime": [],
    }
    seen_runtime_paths = set()
    for line in lines[1:-1]:
        fields = line.split("\t")
        kind = fields[0]
        if kind in {"bootstrap", "checkout-root", "interpreter", "runtime-manifest"}:
            if len(fields) != 7:
                raise RuntimeError("external bootstrap singleton record mismatch")
            key = kind.replace("-", "_")
            if result[key] is not None:
                raise RuntimeError("external bootstrap singleton is duplicated")
            record = {
                "path": lexical_absolute_path(fields[1], kind + " path"),
                "sha256": fields[2],
                "bytes": _parse_integer(fields[3], kind + " bytes"),
                "device": _parse_integer(fields[4], kind + " device"),
                "inode": _parse_integer(fields[5], kind + " inode"),
                "fd": _parse_integer(fields[6], kind + " descriptor"),
            }
            if kind == "checkout-root":
                if record["sha256"] != "-" or record["bytes"] != 0:
                    raise RuntimeError("checkout root attestation mismatch")
            else:
                require_hash(record["sha256"], kind + " hash")
                if record["bytes"] <= 0:
                    raise RuntimeError(kind + " attested bytes mismatch")
            result[key] = record
        elif kind == "source":
            if len(fields) != 8:
                raise RuntimeError("external bootstrap source record mismatch")
            relative = fields[1]
            relative_components(relative, "attested source path")
            if relative in result["sources"]:
                raise RuntimeError("external bootstrap source is duplicated")
            record = {
                "relative_path": relative,
                "path": lexical_absolute_path(fields[2], "attested source path"),
                "sha256": require_hash(fields[3], "attested source hash"),
                "bytes": _parse_integer(fields[4], "attested source bytes"),
                "device": _parse_integer(fields[5], "attested source device"),
                "inode": _parse_integer(fields[6], "attested source inode"),
                "fd": _parse_integer(fields[7], "attested source descriptor"),
            }
            result["sources"][relative] = record
        elif kind == "runtime":
            if len(fields) != 8 or fields[1] not in {
                "bootstrap",
                "python",
                "runtime-held",
                "runtime-inventory",
            }:
                raise RuntimeError("external bootstrap runtime record mismatch")
            path = lexical_absolute_path(fields[2], "attested runtime path")
            if path in seen_runtime_paths:
                raise RuntimeError("external bootstrap runtime path is duplicated")
            seen_runtime_paths.add(path)
            descriptor_text = fields[7]
            descriptor = (
                -1
                if descriptor_text == "-1"
                else _parse_integer(descriptor_text, "attested runtime descriptor")
            )
            if fields[1] == "runtime-held" and descriptor < 0:
                raise RuntimeError("held runtime descriptor is missing")
            if fields[1] != "runtime-held" and descriptor != -1:
                raise RuntimeError(
                    "inventory runtime unexpectedly retained a descriptor"
                )
            result["runtime"].append(
                {
                    "role": fields[1],
                    "path": path,
                    "sha256": require_hash(fields[3], "attested runtime hash"),
                    "bytes": _parse_integer(fields[4], "attested runtime bytes"),
                    "device": _parse_integer(fields[5], "attested runtime device"),
                    "inode": _parse_integer(fields[6], "attested runtime inode"),
                    "fd": descriptor,
                }
            )
        else:
            raise RuntimeError("external bootstrap attestation record kind mismatch")
    if (
        any(
            result[key] is None
            for key in ("bootstrap", "checkout_root", "interpreter", "runtime_manifest")
        )
        or set(result["sources"]) != set(SOURCE_PATHS)
        or not result["runtime"]
    ):
        raise RuntimeError("external bootstrap attestation inventory mismatch")
    return result


def _record_contract(record: dict, payload: bytes, metadata: os.stat_result) -> dict:
    contract = file_contract(record["path"], payload, metadata)
    expected = {
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "device": record["device"],
        "inode": record["inode"],
    }
    observed = {
        key: contract[key] for key in ("path", "bytes", "sha256", "device", "inode")
    }
    if observed != expected:
        raise RuntimeError(
            "attested file descriptor identity mismatch: " + record["path"]
        )
    return contract


def verify_attested_file(record: dict, label: str) -> tuple[bytes, dict]:
    descriptor = record["fd"]
    if type(descriptor) is not int or descriptor < 0:
        raise RuntimeError(label + " retained descriptor mismatch")
    payload, metadata = read_descriptor(descriptor, label)
    contract = _record_contract(record, payload, metadata)
    live_fd = open_absolute_nofollow(record["path"], label)
    try:
        live_payload, live_metadata = read_descriptor(live_fd, label)
    finally:
        os.close(live_fd)
    if live_payload != payload or descriptor_state(live_metadata) != descriptor_state(
        metadata
    ):
        raise RuntimeError(label + " path changed after bootstrap pin")
    return payload, contract


def verify_attested_source(
    checkout_root_fd: int,
    checkout_root_path: str,
    record: dict,
) -> tuple[bytes, dict]:
    payload, metadata = read_descriptor(
        record["fd"], "attested source " + record["relative_path"]
    )
    expected_path = checkout_root_path + "/" + record["relative_path"]
    if record["path"] != expected_path:
        raise RuntimeError("attested source escaped checkout root")
    contract = _record_contract(record, payload, metadata)
    live_fd = open_relative_nofollow(
        checkout_root_fd,
        record["relative_path"],
        "attested source " + record["relative_path"],
    )
    try:
        live_payload, live_metadata = read_descriptor(
            live_fd, "attested source " + record["relative_path"]
        )
    finally:
        os.close(live_fd)
    if live_payload != payload or descriptor_state(live_metadata) != descriptor_state(
        metadata
    ):
        raise RuntimeError(
            "attested source path changed after bootstrap pin: "
            + record["relative_path"]
        )
    return payload, contract


def verify_runtime_inventory(records: list[dict]) -> dict:
    files = {}
    held_fds = {}
    for record in records:
        if record["role"] in {"bootstrap", "python"}:
            continue
        if record["role"] == "runtime-held":
            payload, contract = verify_attested_file(record, "held runtime file")
            held_fds[record["path"]] = record["fd"]
        else:
            descriptor = open_absolute_nofollow(
                record["path"], "runtime inventory file"
            )
            try:
                payload, metadata = read_descriptor(
                    descriptor, "runtime inventory file"
                )
            finally:
                os.close(descriptor)
            contract = _record_contract(record, payload, metadata)
        files[record["path"]] = {
            **contract,
            "retained_descriptor": record["role"] == "runtime-held",
        }
    inventory = {
        "schema": "shohin-ocsc-external-runtime-inventory-v1",
        "files": files,
        "file_count": len(files),
        "held_file_count": len(held_fds),
        "files_sha256": hash_json(files),
    }
    inventory["payload_sha256"] = hash_json(inventory)
    return {"contract": inventory, "held_fds": held_fds}


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run exact approved OCSC bytes")
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--checkout-root", required=True, type=Path)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--prereg-sha256", required=True)
    parser.add_argument("--generator-sha256", required=True)
    parser.add_argument("--tests-sha256", required=True)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("test", "qualification", "production"),
    )
    parser.add_argument("--print-qualification-contract", action="store_true")
    parser.add_argument("generator_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    generator_argv = list(args.generator_argv)
    if generator_argv[:1] == ["--"]:
        generator_argv = generator_argv[1:]
    if args.print_qualification_contract and generator_argv:
        raise RuntimeError(
            "qualification-contract inspection accepts no generator argv"
        )
    if not args.print_qualification_contract and not generator_argv:
        raise RuntimeError("generator argv is empty")
    return args, generator_argv


def attest_external_bootstrap(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str], dict, dict]:
    if not (
        sys.flags.isolated
        and sys.flags.safe_path
        and sys.flags.no_user_site
        and sys.flags.no_site
        and sys.dont_write_bytecode
    ):
        raise RuntimeError("runner requires Python -I -S -B")
    if _EXTERNAL_ATTESTATION_BYTES is None:
        raise RuntimeError(
            "direct runner execution is forbidden; compile and use the external C bootstrap"
        )
    args, generator_argv = parse_args(argv)
    attestation = parse_external_attestation(_EXTERNAL_ATTESTATION_BYTES)
    bootstrap_payload, bootstrap_contract = verify_attested_file(
        attestation["bootstrap"], "external bootstrap executable"
    )
    manifest_payload, manifest_contract = verify_attested_file(
        attestation["runtime_manifest"], "external runtime manifest"
    )
    interpreter_payload, interpreter_contract = verify_attested_file(
        attestation["interpreter"], "Python interpreter"
    )
    root = attestation["checkout_root"]
    root_fd = root["fd"]
    root_live_fd = open_absolute_nofollow(
        root["path"], "attested checkout root", directory=True
    )
    try:
        held_root = os.fstat(root_fd)
        live_root = os.fstat(root_live_fd)
    finally:
        os.close(root_live_fd)
    if (
        not stat.S_ISDIR(held_root.st_mode)
        or (held_root.st_dev, held_root.st_ino) != (root["device"], root["inode"])
        or (live_root.st_dev, live_root.st_ino) != (held_root.st_dev, held_root.st_ino)
    ):
        raise RuntimeError("attested checkout root identity drifted")
    if lexical_absolute_path(args.checkout_root, "checkout root") != root["path"]:
        raise RuntimeError("checkout root argument does not match external attestation")
    source_payloads = {}
    source_contracts = {}
    for relative in SOURCE_PATHS:
        payload, contract = verify_attested_source(
            root_fd, root["path"], attestation["sources"][relative]
        )
        source_payloads[relative] = payload
        source_contracts[relative] = contract
    expected_hashes = {
        relative: require_hash(getattr(args, argument), relative + " hash")
        for relative, argument in HASH_ARGUMENTS.items()
    }
    for relative in SOURCE_PATHS:
        if source_contracts[relative]["sha256"] != expected_hashes[relative]:
            raise RuntimeError(relative + " does not match its approved hash")
    if (
        lexical_absolute_path(args.runner, "runner")
        != source_contracts[RUNNER_RELATIVE_PATH]["path"]
        or _LOADED_RUNNER_PAYLOAD != source_payloads[RUNNER_RELATIVE_PATH]
    ):
        raise RuntimeError("runner path or loaded bytes mismatch")
    if bootstrap_contract["sha256"] != require_hash(
        args.bootstrap_sha256, "bootstrap hash"
    ):
        raise RuntimeError("external bootstrap executable hash mismatch")
    if manifest_contract["sha256"] != require_hash(
        args.runtime_manifest_sha256, "runtime manifest hash"
    ):
        raise RuntimeError("external runtime manifest hash mismatch")
    if interpreter_contract["sha256"] != require_hash(
        args.python_sha256, "Python hash"
    ):
        raise RuntimeError("Python interpreter hash mismatch")
    if (
        lexical_absolute_path(args.python, "Python path")
        != interpreter_contract["path"]
    ):
        raise RuntimeError("Python path does not match external attestation")
    if manifest_payload != os.pread(
        attestation["runtime_manifest"]["fd"], len(manifest_payload), 0
    ):
        raise RuntimeError("runtime manifest descriptor changed")
    runtime = verify_runtime_inventory(attestation["runtime"])
    bound = {
        "attestation": attestation,
        "bootstrap_payload": bootstrap_payload,
        "bootstrap_contract": bootstrap_contract,
        "manifest_payload": manifest_payload,
        "manifest_contract": manifest_contract,
        "interpreter_payload": interpreter_payload,
        "interpreter_contract": interpreter_contract,
        "checkout_root_fd": root_fd,
        "checkout_root_path": root["path"],
        "source_payloads": source_payloads,
        "source_contracts": source_contracts,
        "runtime_inventory": runtime["contract"],
        "held_runtime_fds": runtime["held_fds"],
    }
    return args, generator_argv, attestation, bound


def configure_isolated_distribution_paths() -> None:
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value is None:
            continue
        path = lexical_absolute_path(value, "Python distribution path")
        descriptor = open_absolute_nofollow(
            path, "Python distribution path", directory=True
        )
        os.close(descriptor)
        if path not in sys.path:
            sys.path.append(path)


def qualification_contract() -> dict:
    contract = {
        "schema": "shohin-ocsc-linux-lustre-two-host-qualification-source-v2",
        "status": "unexecuted-source-contract-only",
        "external_bootstrap": {
            "source": RUNNER_RELATIVE_PATH,
            "compile_as": "ISO-C11",
            "python_flags": ["-I", "-S", "-B", "-c"],
            "actual_executable_sha256_required": True,
            "reviewed_runtime_manifest_required": True,
            "runtime_files_attested_before_project_python": True,
            "runner_executed_from_retained_descriptor": True,
            "imported_python_launcher_constant_authorized": False,
            "caller_constructed_context_authorized": False,
        },
        "path_authority": {
            "pinned_checkout_root_descriptor": True,
            "component_by_component_o_nofollow": True,
            "resolve_or_realpath_before_open_authorized": False,
            "preexisting_symlink_alias_authorized": False,
            "post_pin_component_replacement_authorized": False,
        },
        "source_inventory": list(SOURCE_PATHS),
        "runtime_closure_required_before_generator_action": True,
        "inputs": {
            "real_frozen_tokenizer_required": True,
            "real_registry_and_commitments_required": True,
            "synthetic_tokenizer_authorized": False,
            "fixture_only_inputs_authorized": False,
        },
        "filesystem": {
            "linux_required": True,
            "lustre_required": True,
            "same_mount_source_required": True,
            "same_output_parent_device_inode_required": True,
            "same_host_fork_counts_as_cross_node": False,
        },
        "roles": {
            "primary_publisher": "externally bootstrapped Linux host A",
            "secondary_contender_and_restart_observer": (
                "externally bootstrapped distinct Linux host B"
            ),
            "production_broker": "immutable no-replace event transfer path",
            "independent_reviewer": "derives report marker and receipt from events",
            "distinct_host_kernel_identities_required": True,
        },
        "required_checks": list(QUALIFICATION_CHECKS),
        "required_crash_points": list(QUALIFICATION_CRASH_POINTS),
        "executable_actions": {
            "source_manifest_inspection": "--print-source-manifest",
            "publisher_and_restart_path": "--qualification-output-dir",
            "publisher_hard_crash_selector": "--qualification-crash-point",
            "production_broker_transfer": ("--qualification-broker-transfer-event"),
            "event_derived_report_marker_receipt": (
                "--qualification-write-evidence-package"
            ),
        },
        "event_authority": {
            "raw_events_signed_and_hash_chained": True,
            "broker_receipts_signed_and_hash_chained": True,
            "checks_are_event_id_lists_not_booleans": True,
            "counts_mechanically_derived": True,
            "caller_summary_authority": False,
            "caller_check_map_authority": False,
        },
        "evidence_retention": {
            "delete_or_unlink_authorized": False,
            "rewrite_authorized": False,
            "failed_and_partial_stages_retained": True,
            "journals_leases_broker_records_receipts_retained": True,
            "new_attempt_requires_new_output_identity": True,
        },
        "qualification_authority": False,
        "bundle_publication_authority": False,
        "consumer_integration_authority": False,
        "fit_or_evaluation_authority": False,
        "gpu_authority": False,
        "scientific_claim_authority": False,
        "claim_boundary": (
            "source_review_and_local_or_linux_filesystem_qualification_only"
        ),
    }
    contract["payload_sha256"] = hash_json(contract)
    return contract


def _source_snapshot_contract(contract: dict) -> dict:
    return {
        "resolved_path": contract["path"],
        "bytes": contract["bytes"],
        "sha256": contract["sha256"],
        "mode": contract["mode"],
        "owner_uid": contract["owner_uid"],
        "device": contract["device"],
        "inode": contract["inode"],
        "hard_links": contract["hard_links"],
    }


def main() -> None:
    argv = list(sys.argv[1:])
    try:
        args, generator_argv, attestation, bound = attest_external_bootstrap(argv)
        configure_isolated_distribution_paths()
        source_contracts = {
            relative: _source_snapshot_contract(contract)
            for relative, contract in bound["source_contracts"].items()
        }
        interpreter_contract = _source_snapshot_contract(bound["interpreter_contract"])
        bootstrap_identity = {
            "schema": "shohin-ocsc-external-bootstrap-identity-v1",
            "attestation_sha256": attestation["attestation_sha256"],
            "executable": _source_snapshot_contract(bound["bootstrap_contract"]),
            "runtime_manifest": _source_snapshot_contract(bound["manifest_contract"]),
            "runtime_inventory_sha256": bound["runtime_inventory"]["payload_sha256"],
            "checkout_root_path": bound["checkout_root_path"],
            "checkout_root_device": os.fstat(bound["checkout_root_fd"]).st_dev,
            "checkout_root_inode": os.fstat(bound["checkout_root_fd"]).st_ino,
            "authority": False,
        }
        bootstrap_identity["payload_sha256"] = hash_json(bootstrap_identity)
        execution = {
            "schema": "shohin-ocsc-bootstrap-execution-v2",
            "profile": args.profile,
            "source_bound": True,
            "qualification_authority": False,
            "production_authority": False,
            "bootstrap": bootstrap_identity,
            "sources": source_contracts,
            "sources_sha256": hash_json(source_contracts),
            "interpreter": interpreter_contract,
            "external_runtime_inventory": bound["runtime_inventory"],
            "generator_argv": generator_argv,
            "generator_argv_sha256": hash_json(generator_argv),
        }
        execution["payload_sha256"] = hash_json(execution)
        if args.print_qualification_contract:
            print(
                json.dumps(qualification_contract(), ensure_ascii=True, sort_keys=True)
            )
            return
        module_name = "_ocsc_pinned_generator"
        module = types.ModuleType(module_name)
        module.__file__ = source_contracts[GENERATOR_RELATIVE_PATH]["resolved_path"]
        module.__package__ = None
        module.__dict__["_OCSC_BOOTSTRAP_EXECUTION_CONTEXT"] = {
            "contract": execution,
            "checkout_root_fd": bound["checkout_root_fd"],
            "checkout_root_path": bound["checkout_root_path"],
            "source_fds": {
                relative: attestation["sources"][relative]["fd"]
                for relative in SOURCE_PATHS
            },
            "source_snapshots": {
                relative: {
                    "payload": bound["source_payloads"][relative],
                    "contract": source_contracts[relative],
                }
                for relative in SOURCE_PATHS
            },
            "runtime_fds": {
                "bootstrap": attestation["bootstrap"]["fd"],
                "interpreter": attestation["interpreter"]["fd"],
                "runtime_manifest": attestation["runtime_manifest"]["fd"],
                **{
                    "held:" + path: descriptor
                    for path, descriptor in bound["held_runtime_fds"].items()
                },
            },
            "runtime_snapshots": {
                "bootstrap": {
                    "payload": bound["bootstrap_payload"],
                    "contract": bootstrap_identity["executable"],
                },
                "interpreter": {
                    "payload": bound["interpreter_payload"],
                    "contract": interpreter_contract,
                },
                "runtime_manifest": {
                    "payload": bound["manifest_payload"],
                    "contract": bootstrap_identity["runtime_manifest"],
                },
            },
        }
        code = compile(
            bound["source_payloads"][GENERATOR_RELATIVE_PATH],
            source_contracts[GENERATOR_RELATIVE_PATH]["resolved_path"],
            "exec",
            dont_inherit=True,
            optimize=0,
        )
        sys.modules[module_name] = module
        try:
            exec(code, module.__dict__)
            result, source_manifest = module.bootstrap_cli(generator_argv)
            module.bootstrap_execution_contract(generator_argv, required=True)
            module.validate_source_manifest_contract(source_manifest)
        finally:
            if sys.modules.get(module_name) is module:
                del sys.modules[module_name]
        attest_external_bootstrap(argv)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit("OCSC runner rejected: {}".format(error)) from error


if __name__ == "__main__":
    main()
# endif
