/*
 * Standalone EPISODE categorical-machine runtime.
 *
 * This executable deliberately has no dependency on Shohin source code or data
 * formats. It accepts exactly three paths:
 *
 *   episode_functor_runtime_c MACHINE.bin QUERIES.bin TRANSCRIPT.bin
 *
 * All integers are unsigned little-endian. Keys are typed uint64 values:
 * active keys must be nonzero and unique within their state, action, or
 * observer namespace. Inactive slots and every reserved field must be zero.
 *
 * MACHINE.bin (exactly 1536 bytes)
 *   0       8   magic "EFCMACH\0"
 *   8       4   version = 1
 *   12      4   header size = 64
 *   16      4   file size = 1536
 *   20      4   flags = 0
 *   24      2   active state count
 *   26      2   active action count
 *   28      2   active observer count
 *   30      2   reserved = 0
 *   32      8   state-slot active mask (16 slots)
 *   40      8   action-slot active mask (8 slots)
 *   48      8   observer-slot active mask (8 slots)
 *   56      1   initial state slot
 *   57      7   reserved = 0
 *   64    128   16 state keys (uint64 each)
 *   192    64   8 action keys (uint64 each)
 *   256    64   8 observer keys (uint64 each)
 *   320   128   next[action_slot * 16 + state_slot] (uint8 destinations)
 *   448  1024   observer[observer_slot * 16 + state_slot] (uint64 answers)
 *   1472   32   reserved = 0
 *   1504   32   SHA-256 of bytes [0, 1504)
 *
 * QUERIES.bin (64 + query_count * 320 + 32 bytes)
 *   Header:
 *     0      8   magic "EFCQRY\0\0"
 *     8      4   version = 1
 *     12     4   header size = 64
 *     16     4   query record size = 320
 *     20     4   query count (1..100000)
 *     24    32   MACHINE.bin payload hash (the machine's trailing hash)
 *     56     4   flags = 0
 *     60     4   reserved = 0
 *   Each 320-byte record:
 *     0      8   nonzero, file-unique challenge ID
 *     8      8   active state key
 *     16     8   active observer key
 *     24     2   word length (0..32)
 *     26     2   flags = 0
 *     28     4   reserved = 0
 *     32   256   32 action keys (uint64); entries after word length are zero
 *     288   32   reserved = 0
 *   Final 32 bytes are SHA-256 of all preceding query-file bytes.
 *
 * TRANSCRIPT.bin (96 + query_count * 32 + 32 bytes)
 *   Header:
 *     0      8   magic "EFCOUT\0\0"
 *     8      4   version = 1
 *     12     4   header size = 96
 *     16     4   transcript record size = 32
 *     20     4   record count
 *     24    32   MACHINE.bin payload hash
 *     56    32   QUERIES.bin payload hash (the query file's trailing hash)
 *     88     4   flags = 0
 *     92     4   reserved = 0
 *   Each 32-byte record:
 *     0      8   challenge ID
 *     8      8   final state key
 *     16     8   observer answer
 *     24     2   final state slot
 *     26     2   status = 0 (success)
 *     28     2   executed step count
 *     30     2   reserved = 0
 *   Final 32 bytes are SHA-256 of all preceding transcript bytes.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FORMAT_VERSION 1U
#define MACHINE_SIZE 1536U
#define MACHINE_HASH_OFFSET 1504U
#define MACHINE_HEADER_SIZE 64U
#define QUERY_HEADER_SIZE 64U
#define QUERY_RECORD_SIZE 320U
#define TRANSCRIPT_HEADER_SIZE 96U
#define TRANSCRIPT_RECORD_SIZE 32U
#define HASH_SIZE 32U
#define MAX_STATES 16U
#define MAX_ACTIONS 8U
#define MAX_OBSERVERS 8U
#define MAX_WORD 32U
#define MAX_QUERIES 100000U

static const uint8_t MACHINE_MAGIC[8] = {
    'E', 'F', 'C', 'M', 'A', 'C', 'H', '\0'
};
static const uint8_t QUERY_MAGIC[8] = {
    'E', 'F', 'C', 'Q', 'R', 'Y', '\0', '\0'
};
static const uint8_t TRANSCRIPT_MAGIC[8] = {
    'E', 'F', 'C', 'O', 'U', 'T', '\0', '\0'
};

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t block[64];
    size_t block_size;
} Sha256;

typedef struct {
    uint64_t state_keys[MAX_STATES];
    uint64_t action_keys[MAX_ACTIONS];
    uint64_t observer_keys[MAX_OBSERVERS];
    uint8_t next[MAX_ACTIONS * MAX_STATES];
    uint64_t observer[MAX_OBSERVERS * MAX_STATES];
    uint64_t state_mask;
    uint64_t action_mask;
    uint64_t observer_mask;
    uint8_t initial_state;
    uint8_t payload_hash[HASH_SIZE];
} Machine;

static uint32_t rotate_right(uint32_t value, unsigned int amount)
{
    return (value >> amount) | (value << (32U - amount));
}

static void sha256_transform(Sha256 *context, const uint8_t block[64])
{
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
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
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
    size_t index;

    for (index = 0; index < 16U; ++index) {
        size_t offset = index * 4U;
        words[index] = ((uint32_t)block[offset] << 24U)
            | ((uint32_t)block[offset + 1U] << 16U)
            | ((uint32_t)block[offset + 2U] << 8U)
            | (uint32_t)block[offset + 3U];
    }
    for (index = 16U; index < 64U; ++index) {
        uint32_t left = words[index - 15U];
        uint32_t right = words[index - 2U];
        uint32_t sigma0 = rotate_right(left, 7U)
            ^ rotate_right(left, 18U) ^ (left >> 3U);
        uint32_t sigma1 = rotate_right(right, 17U)
            ^ rotate_right(right, 19U) ^ (right >> 10U);
        words[index] = words[index - 16U] + sigma0
            + words[index - 7U] + sigma1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0; index < 64U; ++index) {
        uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U)
            ^ rotate_right(e, 25U);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 = h + sum1 + choice + constants[index]
            + words[index];
        uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U)
            ^ rotate_right(a, 22U);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
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

static void sha256_init(Sha256 *context)
{
    static const uint32_t initial[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };

    memcpy(context->state, initial, sizeof(initial));
    context->bit_count = 0U;
    context->block_size = 0U;
}

static void sha256_update(Sha256 *context, const uint8_t *data, size_t length)
{
    size_t offset = 0U;

    while (offset < length) {
        size_t available = 64U - context->block_size;
        size_t take = length - offset;
        if (take > available) {
            take = available;
        }
        memcpy(context->block + context->block_size, data + offset, take);
        context->block_size += take;
        offset += take;
        if (context->block_size == 64U) {
            sha256_transform(context, context->block);
            context->bit_count += 512U;
            context->block_size = 0U;
        }
    }
}

static void sha256_final(Sha256 *context, uint8_t digest[HASH_SIZE])
{
    uint64_t total_bits = context->bit_count
        + ((uint64_t)context->block_size * 8U);
    size_t index;

    context->block[context->block_size++] = 0x80U;
    if (context->block_size > 56U) {
        while (context->block_size < 64U) {
            context->block[context->block_size++] = 0U;
        }
        sha256_transform(context, context->block);
        context->block_size = 0U;
    }
    while (context->block_size < 56U) {
        context->block[context->block_size++] = 0U;
    }
    for (index = 0U; index < 8U; ++index) {
        context->block[63U - index] = (uint8_t)(total_bits >> (index * 8U));
    }
    sha256_transform(context, context->block);

    for (index = 0U; index < 8U; ++index) {
        digest[index * 4U] = (uint8_t)(context->state[index] >> 24U);
        digest[index * 4U + 1U] =
            (uint8_t)(context->state[index] >> 16U);
        digest[index * 4U + 2U] =
            (uint8_t)(context->state[index] >> 8U);
        digest[index * 4U + 3U] = (uint8_t)context->state[index];
    }
}

static void sha256_bytes(
    const uint8_t *data,
    size_t length,
    uint8_t digest[HASH_SIZE]
)
{
    Sha256 context;
    sha256_init(&context);
    sha256_update(&context, data, length);
    sha256_final(&context, digest);
}

static uint16_t load_u16(const uint8_t *data)
{
    return (uint16_t)((uint16_t)data[0]
        | ((uint16_t)data[1] << 8U));
}

static uint32_t load_u32(const uint8_t *data)
{
    return (uint32_t)data[0]
        | ((uint32_t)data[1] << 8U)
        | ((uint32_t)data[2] << 16U)
        | ((uint32_t)data[3] << 24U);
}

static uint64_t load_u64(const uint8_t *data)
{
    uint64_t value = 0U;
    unsigned int index;
    for (index = 0U; index < 8U; ++index) {
        value |= (uint64_t)data[index] << (index * 8U);
    }
    return value;
}

static void store_u16(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8U);
}

static void store_u32(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8U);
    data[2] = (uint8_t)(value >> 16U);
    data[3] = (uint8_t)(value >> 24U);
}

static void store_u64(uint8_t *data, uint64_t value)
{
    unsigned int index;
    for (index = 0U; index < 8U; ++index) {
        data[index] = (uint8_t)(value >> (index * 8U));
    }
}

static int bytes_are_zero(const uint8_t *data, size_t length)
{
    size_t index;
    uint8_t combined = 0U;
    for (index = 0U; index < length; ++index) {
        combined |= data[index];
    }
    return combined == 0U;
}

static int hashes_equal(
    const uint8_t left[HASH_SIZE],
    const uint8_t right[HASH_SIZE]
)
{
    size_t index;
    uint8_t difference = 0U;
    for (index = 0U; index < HASH_SIZE; ++index) {
        difference |= (uint8_t)(left[index] ^ right[index]);
    }
    return difference == 0U;
}

static unsigned int population_count(uint64_t value)
{
    unsigned int count = 0U;
    while (value != 0U) {
        count += (unsigned int)(value & 1U);
        value >>= 1U;
    }
    return count;
}

static int slot_is_active(uint64_t mask, unsigned int slot)
{
    return ((mask >> slot) & 1U) != 0U;
}

static int keys_are_valid(
    const uint64_t *keys,
    unsigned int capacity,
    uint64_t mask
)
{
    unsigned int left;
    for (left = 0U; left < capacity; ++left) {
        unsigned int right;
        if (!slot_is_active(mask, left)) {
            if (keys[left] != 0U) {
                return 0;
            }
            continue;
        }
        if (keys[left] == 0U) {
            return 0;
        }
        for (right = left + 1U; right < capacity; ++right) {
            if (slot_is_active(mask, right) && keys[left] == keys[right]) {
                return 0;
            }
        }
    }
    return 1;
}

static int find_key(
    const uint64_t *keys,
    unsigned int capacity,
    uint64_t mask,
    uint64_t key
)
{
    unsigned int slot;
    for (slot = 0U; slot < capacity; ++slot) {
        if (slot_is_active(mask, slot) && keys[slot] == key) {
            return (int)slot;
        }
    }
    return -1;
}

static int read_file(
    const char *path,
    size_t maximum_size,
    uint8_t **contents,
    size_t *length
)
{
    FILE *stream = fopen(path, "rb");
    long measured;
    uint8_t *buffer;

    if (stream == NULL) {
        fprintf(stderr, "error: cannot open input '%s': %s\n",
                path, strerror(errno));
        return 0;
    }
    if (fseek(stream, 0L, SEEK_END) != 0) {
        fprintf(stderr, "error: cannot seek input '%s'\n", path);
        fclose(stream);
        return 0;
    }
    measured = ftell(stream);
    if (measured < 0L || (uint64_t)measured > (uint64_t)maximum_size) {
        fprintf(stderr, "error: invalid or excessive input length for '%s'\n",
                path);
        fclose(stream);
        return 0;
    }
    if (fseek(stream, 0L, SEEK_SET) != 0) {
        fprintf(stderr, "error: cannot rewind input '%s'\n", path);
        fclose(stream);
        return 0;
    }
    buffer = (uint8_t *)malloc((size_t)measured == 0U
        ? 1U : (size_t)measured);
    if (buffer == NULL) {
        fprintf(stderr, "error: memory allocation failed\n");
        fclose(stream);
        return 0;
    }
    if ((size_t)measured != 0U
        && fread(buffer, 1U, (size_t)measured, stream) != (size_t)measured) {
        fprintf(stderr, "error: cannot read complete input '%s'\n", path);
        free(buffer);
        fclose(stream);
        return 0;
    }
    if (fclose(stream) != 0) {
        fprintf(stderr, "error: cannot close input '%s'\n", path);
        free(buffer);
        return 0;
    }
    *contents = buffer;
    *length = (size_t)measured;
    return 1;
}

static int parse_machine(
    const uint8_t *data,
    size_t length,
    Machine *machine
)
{
    uint8_t digest[HASH_SIZE];
    unsigned int state;
    unsigned int action;
    unsigned int observer;

    if (length != MACHINE_SIZE) {
        fprintf(stderr, "error: machine length must be exactly %u bytes\n",
                MACHINE_SIZE);
        return 0;
    }
    if (memcmp(data, MACHINE_MAGIC, sizeof(MACHINE_MAGIC)) != 0) {
        fprintf(stderr, "error: machine magic is invalid\n");
        return 0;
    }
    if (load_u32(data + 8U) != FORMAT_VERSION) {
        fprintf(stderr, "error: machine version is unsupported\n");
        return 0;
    }
    if (load_u32(data + 12U) != MACHINE_HEADER_SIZE
        || load_u32(data + 16U) != MACHINE_SIZE) {
        fprintf(stderr, "error: machine declared sizes are invalid\n");
        return 0;
    }
    if (load_u32(data + 20U) != 0U
        || load_u16(data + 30U) != 0U
        || !bytes_are_zero(data + 57U, 7U)
        || !bytes_are_zero(data + 1472U, 32U)) {
        fprintf(stderr, "error: machine flags or padding are nonzero\n");
        return 0;
    }

    machine->state_mask = load_u64(data + 32U);
    machine->action_mask = load_u64(data + 40U);
    machine->observer_mask = load_u64(data + 48U);
    if ((machine->state_mask >> MAX_STATES) != 0U
        || (machine->action_mask >> MAX_ACTIONS) != 0U
        || (machine->observer_mask >> MAX_OBSERVERS) != 0U
        || load_u16(data + 24U)
            != population_count(machine->state_mask)
        || load_u16(data + 26U)
            != population_count(machine->action_mask)
        || load_u16(data + 28U)
            != population_count(machine->observer_mask)
        || load_u16(data + 24U) == 0U
        || load_u16(data + 26U) == 0U
        || load_u16(data + 28U) == 0U) {
        fprintf(stderr, "error: machine active masks or counts are invalid\n");
        return 0;
    }

    machine->initial_state = data[56U];
    if (machine->initial_state >= MAX_STATES
        || !slot_is_active(machine->state_mask, machine->initial_state)) {
        fprintf(stderr, "error: machine initial state is inactive\n");
        return 0;
    }

    for (state = 0U; state < MAX_STATES; ++state) {
        machine->state_keys[state] = load_u64(data + 64U + state * 8U);
    }
    for (action = 0U; action < MAX_ACTIONS; ++action) {
        machine->action_keys[action] = load_u64(
            data + 192U + action * 8U
        );
    }
    for (observer = 0U; observer < MAX_OBSERVERS; ++observer) {
        machine->observer_keys[observer] = load_u64(
            data + 256U + observer * 8U
        );
    }
    if (!keys_are_valid(
            machine->state_keys, MAX_STATES, machine->state_mask)
        || !keys_are_valid(
            machine->action_keys, MAX_ACTIONS, machine->action_mask)
        || !keys_are_valid(
            machine->observer_keys, MAX_OBSERVERS, machine->observer_mask)) {
        fprintf(stderr, "error: machine keys are zero, duplicate, or padded\n");
        return 0;
    }

    for (action = 0U; action < MAX_ACTIONS; ++action) {
        for (state = 0U; state < MAX_STATES; ++state) {
            unsigned int offset = action * MAX_STATES + state;
            uint8_t destination = data[320U + offset];
            int active = slot_is_active(machine->action_mask, action)
                && slot_is_active(machine->state_mask, state);
            if (!active && destination != 0U) {
                fprintf(stderr, "error: machine transition padding is nonzero\n");
                return 0;
            }
            if (active && (destination >= MAX_STATES
                || !slot_is_active(machine->state_mask, destination))) {
                fprintf(stderr, "error: machine transition destination is invalid\n");
                return 0;
            }
            machine->next[offset] = destination;
        }
    }

    for (observer = 0U; observer < MAX_OBSERVERS; ++observer) {
        for (state = 0U; state < MAX_STATES; ++state) {
            unsigned int offset = observer * MAX_STATES + state;
            uint64_t answer = load_u64(data + 448U + offset * 8U);
            int active = slot_is_active(machine->observer_mask, observer)
                && slot_is_active(machine->state_mask, state);
            if (!active && answer != 0U) {
                fprintf(stderr, "error: machine observer padding is nonzero\n");
                return 0;
            }
            machine->observer[offset] = answer;
        }
    }

    sha256_bytes(data, MACHINE_HASH_OFFSET, digest);
    if (!hashes_equal(digest, data + MACHINE_HASH_OFFSET)) {
        fprintf(stderr, "error: machine hash mismatch\n");
        return 0;
    }
    memcpy(machine->payload_hash, digest, HASH_SIZE);
    return 1;
}

static int compare_u64(const void *left, const void *right)
{
    uint64_t left_value = *(const uint64_t *)left;
    uint64_t right_value = *(const uint64_t *)right;
    if (left_value < right_value) {
        return -1;
    }
    if (left_value > right_value) {
        return 1;
    }
    return 0;
}

static int validate_queries(
    const uint8_t *data,
    size_t length,
    const Machine *machine,
    uint32_t *query_count
)
{
    uint8_t digest[HASH_SIZE];
    uint32_t count;
    size_t expected_length;
    uint64_t *identifiers;
    uint32_t query_index;

    if (length < QUERY_HEADER_SIZE + HASH_SIZE) {
        fprintf(stderr, "error: query file length is invalid\n");
        return 0;
    }
    if (memcmp(data, QUERY_MAGIC, sizeof(QUERY_MAGIC)) != 0) {
        fprintf(stderr, "error: query magic is invalid\n");
        return 0;
    }
    if (load_u32(data + 8U) != FORMAT_VERSION) {
        fprintf(stderr, "error: query version is unsupported\n");
        return 0;
    }
    if (load_u32(data + 12U) != QUERY_HEADER_SIZE
        || load_u32(data + 16U) != QUERY_RECORD_SIZE) {
        fprintf(stderr, "error: query declared sizes are invalid\n");
        return 0;
    }
    count = load_u32(data + 20U);
    if (count == 0U || count > MAX_QUERIES) {
        fprintf(stderr, "error: query count is invalid\n");
        return 0;
    }
    expected_length = QUERY_HEADER_SIZE
        + (size_t)count * QUERY_RECORD_SIZE + HASH_SIZE;
    if (length != expected_length) {
        fprintf(stderr, "error: query file length does not match count\n");
        return 0;
    }
    if (!hashes_equal(data + 24U, machine->payload_hash)) {
        fprintf(stderr, "error: query machine hash does not match\n");
        return 0;
    }
    if (load_u32(data + 56U) != 0U
        || load_u32(data + 60U) != 0U) {
        fprintf(stderr, "error: query flags or header padding are nonzero\n");
        return 0;
    }
    sha256_bytes(data, length - HASH_SIZE, digest);
    if (!hashes_equal(digest, data + length - HASH_SIZE)) {
        fprintf(stderr, "error: query hash mismatch\n");
        return 0;
    }

    identifiers = (uint64_t *)malloc((size_t)count * sizeof(uint64_t));
    if (identifiers == NULL) {
        fprintf(stderr, "error: memory allocation failed\n");
        return 0;
    }
    for (query_index = 0U; query_index < count; ++query_index) {
        const uint8_t *record = data + QUERY_HEADER_SIZE
            + (size_t)query_index * QUERY_RECORD_SIZE;
        uint64_t identifier = load_u64(record);
        uint16_t word_length = load_u16(record + 24U);
        unsigned int word_index;

        identifiers[query_index] = identifier;
        if (identifier == 0U) {
            fprintf(stderr, "error: query challenge ID is zero\n");
            free(identifiers);
            return 0;
        }
        if (find_key(machine->state_keys, MAX_STATES,
                     machine->state_mask, load_u64(record + 8U)) < 0) {
            fprintf(stderr, "error: query state key is unknown\n");
            free(identifiers);
            return 0;
        }
        if (find_key(machine->observer_keys, MAX_OBSERVERS,
                     machine->observer_mask, load_u64(record + 16U)) < 0) {
            fprintf(stderr, "error: query observer key is unknown\n");
            free(identifiers);
            return 0;
        }
        if (word_length > MAX_WORD) {
            fprintf(stderr, "error: query word length is invalid\n");
            free(identifiers);
            return 0;
        }
        if (load_u16(record + 26U) != 0U
            || load_u32(record + 28U) != 0U
            || !bytes_are_zero(record + 288U, 32U)) {
            fprintf(stderr, "error: query record flags or padding are nonzero\n");
            free(identifiers);
            return 0;
        }
        for (word_index = 0U; word_index < MAX_WORD; ++word_index) {
            uint64_t action_key = load_u64(
                record + 32U + word_index * 8U
            );
            if (word_index < word_length) {
                if (find_key(machine->action_keys, MAX_ACTIONS,
                             machine->action_mask, action_key) < 0) {
                    fprintf(stderr, "error: query action key is unknown\n");
                    free(identifiers);
                    return 0;
                }
            } else if (action_key != 0U) {
                fprintf(stderr, "error: query action padding is nonzero\n");
                free(identifiers);
                return 0;
            }
        }
    }

    qsort(identifiers, count, sizeof(uint64_t), compare_u64);
    for (query_index = 1U; query_index < count; ++query_index) {
        if (identifiers[query_index - 1U] == identifiers[query_index]) {
            fprintf(stderr, "error: query challenge IDs are duplicate\n");
            free(identifiers);
            return 0;
        }
    }
    free(identifiers);
    *query_count = count;
    return 1;
}

static int build_transcript(
    const Machine *machine,
    const uint8_t *queries,
    size_t query_length,
    uint32_t query_count,
    uint8_t **transcript,
    size_t *transcript_length
)
{
    size_t length = TRANSCRIPT_HEADER_SIZE
        + (size_t)query_count * TRANSCRIPT_RECORD_SIZE + HASH_SIZE;
    uint8_t *output = (uint8_t *)calloc(length, 1U);
    uint32_t query_index;

    if (output == NULL) {
        fprintf(stderr, "error: memory allocation failed\n");
        return 0;
    }
    memcpy(output, TRANSCRIPT_MAGIC, sizeof(TRANSCRIPT_MAGIC));
    store_u32(output + 8U, FORMAT_VERSION);
    store_u32(output + 12U, TRANSCRIPT_HEADER_SIZE);
    store_u32(output + 16U, TRANSCRIPT_RECORD_SIZE);
    store_u32(output + 20U, query_count);
    memcpy(output + 24U, machine->payload_hash, HASH_SIZE);
    memcpy(output + 56U, queries + query_length - HASH_SIZE, HASH_SIZE);

    for (query_index = 0U; query_index < query_count; ++query_index) {
        const uint8_t *query = queries + QUERY_HEADER_SIZE
            + (size_t)query_index * QUERY_RECORD_SIZE;
        uint8_t *record = output + TRANSCRIPT_HEADER_SIZE
            + (size_t)query_index * TRANSCRIPT_RECORD_SIZE;
        uint16_t word_length = load_u16(query + 24U);
        int state = find_key(machine->state_keys, MAX_STATES,
                             machine->state_mask, load_u64(query + 8U));
        int observer = find_key(
            machine->observer_keys, MAX_OBSERVERS,
            machine->observer_mask, load_u64(query + 16U)
        );
        unsigned int word_index;

        for (word_index = 0U; word_index < word_length; ++word_index) {
            int action = find_key(
                machine->action_keys, MAX_ACTIONS,
                machine->action_mask,
                load_u64(query + 32U + word_index * 8U)
            );
            state = machine->next[
                (unsigned int)action * MAX_STATES + (unsigned int)state
            ];
        }
        store_u64(record, load_u64(query));
        store_u64(record + 8U, machine->state_keys[(unsigned int)state]);
        store_u64(
            record + 16U,
            machine->observer[
                (unsigned int)observer * MAX_STATES + (unsigned int)state
            ]
        );
        store_u16(record + 24U, (uint16_t)state);
        store_u16(record + 26U, 0U);
        store_u16(record + 28U, word_length);
    }
    sha256_bytes(output, length - HASH_SIZE, output + length - HASH_SIZE);
    *transcript = output;
    *transcript_length = length;
    return 1;
}

static int write_output(
    const char *path,
    const uint8_t *data,
    size_t length
)
{
    FILE *stream = fopen(path, "wbx");
    if (stream == NULL) {
        fprintf(stderr, "error: cannot create output '%s': %s\n",
                path, strerror(errno));
        return 0;
    }
    if (fwrite(data, 1U, length, stream) != length) {
        fprintf(stderr, "error: cannot write complete output '%s'\n", path);
        fclose(stream);
        remove(path);
        return 0;
    }
    if (fflush(stream) != 0 || fclose(stream) != 0) {
        fprintf(stderr, "error: cannot finalize output '%s'\n", path);
        remove(path);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    const size_t maximum_query_size = QUERY_HEADER_SIZE
        + (size_t)MAX_QUERIES * QUERY_RECORD_SIZE + HASH_SIZE;
    uint8_t *machine_bytes = NULL;
    uint8_t *query_bytes = NULL;
    uint8_t *transcript = NULL;
    size_t machine_length = 0U;
    size_t query_length = 0U;
    size_t transcript_length = 0U;
    Machine machine;
    uint32_t query_count = 0U;
    int success = 0;

    if (argc != 4) {
        fprintf(stderr,
                "usage: %s MACHINE.bin QUERIES.bin TRANSCRIPT.bin\n",
                argc > 0 ? argv[0] : "episode_functor_runtime_c");
        return 2;
    }
    if (strcmp(argv[1], argv[3]) == 0 || strcmp(argv[2], argv[3]) == 0) {
        fprintf(stderr, "error: output path must differ from input paths\n");
        return 2;
    }
    memset(&machine, 0, sizeof(machine));
    if (!read_file(argv[1], MACHINE_SIZE + 1U,
                   &machine_bytes, &machine_length)
        || !parse_machine(machine_bytes, machine_length, &machine)
        || !read_file(argv[2], maximum_query_size + 1U,
                      &query_bytes, &query_length)
        || !validate_queries(
            query_bytes, query_length, &machine, &query_count)
        || !build_transcript(
            &machine, query_bytes, query_length, query_count,
            &transcript, &transcript_length)
        || !write_output(argv[3], transcript, transcript_length)) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(transcript);
    free(query_bytes);
    free(machine_bytes);
    return success ? 0 : 1;
}
