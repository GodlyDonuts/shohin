//! Standalone EPISODE categorical-machine runtime.
//!
//! This program independently implements the fixed-width wire contract
//! documented in `episode_functor_runtime_c.c`:
//!
//!     episode_functor_runtime_rust MACHINE.bin QUERIES.bin TRANSCRIPT.bin
//!
//! It deliberately has no Shohin dependencies. Unlike the C runtime's flat
//! integer transition update, execution represents the current state as a
//! one-hot bitset. Each action is parsed into a Boolean relation, and a step is
//! the relational image of that bitset. Determinism is checked by requiring
//! every image to remain one-hot.

use std::collections::HashSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

const FORMAT_VERSION: u32 = 1;
const MACHINE_SIZE: usize = 1_536;
const MACHINE_HASH_OFFSET: usize = 1_504;
const MACHINE_HEADER_SIZE: u32 = 64;
const QUERY_HEADER_SIZE: usize = 64;
const QUERY_RECORD_SIZE: usize = 320;
const TRANSCRIPT_HEADER_SIZE: usize = 96;
const TRANSCRIPT_RECORD_SIZE: usize = 32;
const HASH_SIZE: usize = 32;
const MAX_STATES: usize = 16;
const MAX_ACTIONS: usize = 8;
const MAX_OBSERVERS: usize = 8;
const MAX_WORD: usize = 32;
const MAX_QUERIES: usize = 100_000;

const MACHINE_MAGIC: &[u8; 8] = b"EFCMACH\0";
const QUERY_MAGIC: &[u8; 8] = b"EFCQRY\0\0";
const TRANSCRIPT_MAGIC: &[u8; 8] = b"EFCOUT\0\0";

type Digest = [u8; HASH_SIZE];
type Relation = [u16; MAX_STATES];

#[derive(Clone)]
struct Machine {
    state_keys: [u64; MAX_STATES],
    action_keys: [u64; MAX_ACTIONS],
    observer_keys: [u64; MAX_OBSERVERS],
    relations: [Relation; MAX_ACTIONS],
    observer_values: [[u64; MAX_STATES]; MAX_OBSERVERS],
    state_mask: u16,
    action_mask: u8,
    observer_mask: u8,
    payload_hash: Digest,
}

struct Query {
    challenge_id: u64,
    start_slot: usize,
    observer_slot: usize,
    action_slots: Vec<usize>,
}

struct QueryFile {
    records: Vec<Query>,
    payload_hash: Digest,
}

struct Sha256 {
    state: [u32; 8],
    bit_count: u64,
    block: [u8; 64],
    block_size: usize,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            bit_count: 0,
            block: [0; 64],
            block_size: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        let mut offset = 0;
        while offset < data.len() {
            let take = (64 - self.block_size).min(data.len() - offset);
            self.block[self.block_size..self.block_size + take]
                .copy_from_slice(&data[offset..offset + take]);
            self.block_size += take;
            offset += take;
            if self.block_size == 64 {
                let block = self.block;
                self.transform(&block);
                self.bit_count += 512;
                self.block_size = 0;
            }
        }
    }

    fn finish(mut self) -> Digest {
        let total_bits = self.bit_count + (self.block_size as u64) * 8;
        self.block[self.block_size] = 0x80;
        self.block_size += 1;
        if self.block_size > 56 {
            self.block[self.block_size..].fill(0);
            let block = self.block;
            self.transform(&block);
            self.block_size = 0;
        }
        self.block[self.block_size..56].fill(0);
        self.block[56..64].copy_from_slice(&total_bits.to_be_bytes());
        let block = self.block;
        self.transform(&block);

        let mut digest = [0; HASH_SIZE];
        for (index, word) in self.state.iter().enumerate() {
            digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn transform(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a_2f98,
            0x7137_4491,
            0xb5c0_fbcf,
            0xe9b5_dba5,
            0x3956_c25b,
            0x59f1_11f1,
            0x923f_82a4,
            0xab1c_5ed5,
            0xd807_aa98,
            0x1283_5b01,
            0x2431_85be,
            0x550c_7dc3,
            0x72be_5d74,
            0x80de_b1fe,
            0x9bdc_06a7,
            0xc19b_f174,
            0xe49b_69c1,
            0xefbe_4786,
            0x0fc1_9dc6,
            0x240c_a1cc,
            0x2de9_2c6f,
            0x4a74_84aa,
            0x5cb0_a9dc,
            0x76f9_88da,
            0x983e_5152,
            0xa831_c66d,
            0xb003_27c8,
            0xbf59_7fc7,
            0xc6e0_0bf3,
            0xd5a7_9147,
            0x06ca_6351,
            0x1429_2967,
            0x27b7_0a85,
            0x2e1b_2138,
            0x4d2c_6dfc,
            0x5338_0d13,
            0x650a_7354,
            0x766a_0abb,
            0x81c2_c92e,
            0x9272_2c85,
            0xa2bf_e8a1,
            0xa81a_664b,
            0xc24b_8b70,
            0xc76c_51a3,
            0xd192_e819,
            0xd699_0624,
            0xf40e_3585,
            0x106a_a070,
            0x19a4_c116,
            0x1e37_6c08,
            0x2748_774c,
            0x34b0_bcb5,
            0x391c_0cb3,
            0x4ed8_aa4a,
            0x5b9c_ca4f,
            0x682e_6ff3,
            0x748f_82ee,
            0x78a5_636f,
            0x84c8_7814,
            0x8cc7_0208,
            0x90be_fffa,
            0xa450_6ceb,
            0xbef9_a3f7,
            0xc671_78f2,
        ];
        let mut schedule = [0_u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes(chunk.try_into().expect("four-byte SHA-256 word"));
        }
        for index in 16..64 {
            let left = schedule[index - 15];
            let right = schedule[index - 2];
            let sigma0 = left.rotate_right(7) ^ left.rotate_right(18) ^ (left >> 3);
            let sigma1 = right.rotate_right(17) ^ right.rotate_right(19) ^ (right >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(sigma0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(sigma1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        for (target, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *target = target.wrapping_add(value);
        }
    }
}

fn sha256(data: &[u8]) -> Digest {
    let mut context = Sha256::new();
    context.update(data);
    context.finish()
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, String> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| "truncated uint16 field".to_owned())?;
    Ok(u16::from_le_bytes(
        bytes.try_into().map_err(|_| "invalid uint16 field")?,
    ))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated uint32 field".to_owned())?;
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| "invalid uint32 field")?,
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, String> {
    let bytes = data
        .get(offset..offset + 8)
        .ok_or_else(|| "truncated uint64 field".to_owned())?;
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| "invalid uint64 field")?,
    ))
}

fn write_u16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn all_zero(data: &[u8]) -> bool {
    data.iter().all(|byte| *byte == 0)
}

fn active_u16(mask: u16, slot: usize) -> bool {
    mask & (1_u16 << slot) != 0
}

fn active_u8(mask: u8, slot: usize) -> bool {
    mask & (1_u8 << slot) != 0
}

fn validate_keys<const N: usize>(keys: &[u64; N], is_active: impl Fn(usize) -> bool) -> bool {
    let mut seen = HashSet::with_capacity(N);
    for (slot, key) in keys.iter().copied().enumerate() {
        if is_active(slot) {
            if key == 0 || !seen.insert(key) {
                return false;
            }
        } else if key != 0 {
            return false;
        }
    }
    true
}

fn find_key<const N: usize>(
    keys: &[u64; N],
    is_active: impl Fn(usize) -> bool,
    wanted: u64,
) -> Option<usize> {
    keys.iter()
        .enumerate()
        .find_map(|(slot, key)| (is_active(slot) && *key == wanted).then_some(slot))
}

fn one_hot(slot: usize) -> u16 {
    1_u16 << slot
}

fn singleton_slot(bits: u16) -> Option<usize> {
    (bits.count_ones() == 1).then_some(bits.trailing_zeros() as usize)
}

fn relation_image(relation: &Relation, source_set: u16) -> u16 {
    relation
        .iter()
        .enumerate()
        .filter(|(source, _)| source_set & one_hot(*source) != 0)
        .fold(0_u16, |image, (_, destinations)| image | destinations)
}

fn parse_machine(data: &[u8]) -> Result<Machine, String> {
    if data.len() != MACHINE_SIZE {
        return Err(format!(
            "machine length must be exactly {MACHINE_SIZE} bytes"
        ));
    }
    if data.get(..8) != Some(MACHINE_MAGIC.as_slice()) {
        return Err("machine magic is invalid".to_owned());
    }
    if read_u32(data, 8)? != FORMAT_VERSION {
        return Err("machine version is unsupported".to_owned());
    }
    if read_u32(data, 12)? != MACHINE_HEADER_SIZE || read_u32(data, 16)? as usize != MACHINE_SIZE {
        return Err("machine declared sizes are invalid".to_owned());
    }
    if read_u32(data, 20)? != 0
        || read_u16(data, 30)? != 0
        || !all_zero(&data[57..64])
        || !all_zero(&data[1472..1504])
    {
        return Err("machine flags or padding are nonzero".to_owned());
    }

    let state_mask_raw = read_u64(data, 32)?;
    let action_mask_raw = read_u64(data, 40)?;
    let observer_mask_raw = read_u64(data, 48)?;
    if state_mask_raw >> MAX_STATES != 0
        || action_mask_raw >> MAX_ACTIONS != 0
        || observer_mask_raw >> MAX_OBSERVERS != 0
        || state_mask_raw == 0
        || action_mask_raw == 0
        || observer_mask_raw == 0
        || read_u16(data, 24)? as u32 != state_mask_raw.count_ones()
        || read_u16(data, 26)? as u32 != action_mask_raw.count_ones()
        || read_u16(data, 28)? as u32 != observer_mask_raw.count_ones()
    {
        return Err("machine active masks or counts are invalid".to_owned());
    }
    let state_mask = state_mask_raw as u16;
    let action_mask = action_mask_raw as u8;
    let observer_mask = observer_mask_raw as u8;

    let initial_state = data[56] as usize;
    if initial_state >= MAX_STATES || !active_u16(state_mask, initial_state) {
        return Err("machine initial state is inactive".to_owned());
    }

    let mut state_keys = [0_u64; MAX_STATES];
    let mut action_keys = [0_u64; MAX_ACTIONS];
    let mut observer_keys = [0_u64; MAX_OBSERVERS];
    for (slot, key) in state_keys.iter_mut().enumerate() {
        *key = read_u64(data, 64 + slot * 8)?;
    }
    for (slot, key) in action_keys.iter_mut().enumerate() {
        *key = read_u64(data, 192 + slot * 8)?;
    }
    for (slot, key) in observer_keys.iter_mut().enumerate() {
        *key = read_u64(data, 256 + slot * 8)?;
    }
    if !validate_keys(&state_keys, |slot| active_u16(state_mask, slot))
        || !validate_keys(&action_keys, |slot| active_u8(action_mask, slot))
        || !validate_keys(&observer_keys, |slot| active_u8(observer_mask, slot))
    {
        return Err("machine keys are zero, duplicate, or padded".to_owned());
    }

    let mut relations = [[0_u16; MAX_STATES]; MAX_ACTIONS];
    for (action, relation) in relations.iter_mut().enumerate() {
        for (state, destinations) in relation.iter_mut().enumerate() {
            let destination = data[320 + action * MAX_STATES + state] as usize;
            let live_cell = active_u8(action_mask, action) && active_u16(state_mask, state);
            if !live_cell && destination != 0 {
                return Err("machine transition padding is nonzero".to_owned());
            }
            if live_cell {
                if destination >= MAX_STATES || !active_u16(state_mask, destination) {
                    return Err("machine transition destination is invalid".to_owned());
                }
                *destinations = one_hot(destination);
            }
        }
    }

    let mut observer_values = [[0_u64; MAX_STATES]; MAX_OBSERVERS];
    for (observer, row) in observer_values.iter_mut().enumerate() {
        for (state, answer) in row.iter_mut().enumerate() {
            *answer = read_u64(data, 448 + (observer * MAX_STATES + state) * 8)?;
            let live_cell = active_u8(observer_mask, observer) && active_u16(state_mask, state);
            if !live_cell && *answer != 0 {
                return Err("machine observer padding is nonzero".to_owned());
            }
        }
    }

    let payload_hash = sha256(&data[..MACHINE_HASH_OFFSET]);
    if data[MACHINE_HASH_OFFSET..] != payload_hash {
        return Err("machine hash mismatch".to_owned());
    }
    Ok(Machine {
        state_keys,
        action_keys,
        observer_keys,
        relations,
        observer_values,
        state_mask,
        action_mask,
        observer_mask,
        payload_hash,
    })
}

fn parse_queries(data: &[u8], machine: &Machine) -> Result<QueryFile, String> {
    if data.len() < QUERY_HEADER_SIZE + HASH_SIZE {
        return Err("query file length is invalid".to_owned());
    }
    if data.get(..8) != Some(QUERY_MAGIC.as_slice()) {
        return Err("query magic is invalid".to_owned());
    }
    if read_u32(data, 8)? != FORMAT_VERSION {
        return Err("query version is unsupported".to_owned());
    }
    if read_u32(data, 12)? as usize != QUERY_HEADER_SIZE
        || read_u32(data, 16)? as usize != QUERY_RECORD_SIZE
    {
        return Err("query declared sizes are invalid".to_owned());
    }
    let count = read_u32(data, 20)? as usize;
    if count == 0 || count > MAX_QUERIES {
        return Err("query count is invalid".to_owned());
    }
    let expected_length = QUERY_HEADER_SIZE
        .checked_add(
            count
                .checked_mul(QUERY_RECORD_SIZE)
                .ok_or_else(|| "query file length overflows".to_owned())?,
        )
        .and_then(|value| value.checked_add(HASH_SIZE))
        .ok_or_else(|| "query file length overflows".to_owned())?;
    if data.len() != expected_length {
        return Err("query file length does not match count".to_owned());
    }
    if data[24..56] != machine.payload_hash {
        return Err("query machine hash does not match".to_owned());
    }
    if read_u32(data, 56)? != 0 || read_u32(data, 60)? != 0 {
        return Err("query flags or header padding are nonzero".to_owned());
    }
    let payload_hash = sha256(&data[..data.len() - HASH_SIZE]);
    if data[data.len() - HASH_SIZE..] != payload_hash {
        return Err("query hash mismatch".to_owned());
    }

    let mut challenge_ids = HashSet::with_capacity(count);
    let mut records = Vec::with_capacity(count);
    for query_index in 0..count {
        let offset = QUERY_HEADER_SIZE + query_index * QUERY_RECORD_SIZE;
        let record = &data[offset..offset + QUERY_RECORD_SIZE];
        let challenge_id = read_u64(record, 0)?;
        if challenge_id == 0 {
            return Err("query challenge ID is zero".to_owned());
        }
        if !challenge_ids.insert(challenge_id) {
            return Err("query challenge IDs are duplicate".to_owned());
        }
        let start_slot = find_key(
            &machine.state_keys,
            |slot| active_u16(machine.state_mask, slot),
            read_u64(record, 8)?,
        )
        .ok_or_else(|| "query state key is unknown".to_owned())?;
        let observer_slot = find_key(
            &machine.observer_keys,
            |slot| active_u8(machine.observer_mask, slot),
            read_u64(record, 16)?,
        )
        .ok_or_else(|| "query observer key is unknown".to_owned())?;
        let word_length = read_u16(record, 24)? as usize;
        if word_length > MAX_WORD {
            return Err("query word length is invalid".to_owned());
        }
        if read_u16(record, 26)? != 0 || read_u32(record, 28)? != 0 || !all_zero(&record[288..320])
        {
            return Err("query record flags or padding are nonzero".to_owned());
        }
        let mut action_slots = Vec::with_capacity(word_length);
        for word_index in 0..MAX_WORD {
            let key = read_u64(record, 32 + word_index * 8)?;
            if word_index < word_length {
                let action_slot = find_key(
                    &machine.action_keys,
                    |slot| active_u8(machine.action_mask, slot),
                    key,
                )
                .ok_or_else(|| "query action key is unknown".to_owned())?;
                action_slots.push(action_slot);
            } else if key != 0 {
                return Err("query action padding is nonzero".to_owned());
            }
        }
        records.push(Query {
            challenge_id,
            start_slot,
            observer_slot,
            action_slots,
        });
    }
    Ok(QueryFile {
        records,
        payload_hash,
    })
}

fn build_transcript(machine: &Machine, queries: &QueryFile) -> Result<Vec<u8>, String> {
    let length =
        TRANSCRIPT_HEADER_SIZE + queries.records.len() * TRANSCRIPT_RECORD_SIZE + HASH_SIZE;
    let mut output = vec![0_u8; length];
    output[..8].copy_from_slice(TRANSCRIPT_MAGIC);
    write_u32(&mut output, 8, FORMAT_VERSION);
    write_u32(&mut output, 12, TRANSCRIPT_HEADER_SIZE as u32);
    write_u32(&mut output, 16, TRANSCRIPT_RECORD_SIZE as u32);
    write_u32(&mut output, 20, queries.records.len() as u32);
    output[24..56].copy_from_slice(&machine.payload_hash);
    output[56..88].copy_from_slice(&queries.payload_hash);

    for (query_index, query) in queries.records.iter().enumerate() {
        let mut state_set = one_hot(query.start_slot);
        for action_slot in &query.action_slots {
            state_set = relation_image(&machine.relations[*action_slot], state_set);
            if singleton_slot(state_set).is_none() {
                return Err("action relation image is not one-hot".to_owned());
            }
        }
        let final_slot =
            singleton_slot(state_set).ok_or_else(|| "initial state is not one-hot".to_owned())?;
        let offset = TRANSCRIPT_HEADER_SIZE + query_index * TRANSCRIPT_RECORD_SIZE;
        write_u64(&mut output, offset, query.challenge_id);
        write_u64(&mut output, offset + 8, machine.state_keys[final_slot]);
        write_u64(
            &mut output,
            offset + 16,
            machine.observer_values[query.observer_slot][final_slot],
        );
        write_u16(&mut output, offset + 24, final_slot as u16);
        write_u16(&mut output, offset + 26, 0);
        write_u16(&mut output, offset + 28, query.action_slots.len() as u16);
    }
    let digest = sha256(&output[..length - HASH_SIZE]);
    output[length - HASH_SIZE..].copy_from_slice(&digest);
    Ok(output)
}

fn read_bounded(path: &Path, maximum_size: usize) -> Result<Vec<u8>, String> {
    let data = fs::read(path)
        .map_err(|error| format!("cannot read input '{}': {error}", path.display()))?;
    if data.len() > maximum_size {
        return Err(format!(
            "invalid or excessive input length for '{}'",
            path.display()
        ));
    }
    Ok(data)
}

fn write_new(path: &Path, data: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create output '{}': {error}", path.display()))?;
    if let Err(error) = file.write_all(data).and_then(|()| file.flush()) {
        let _ = fs::remove_file(path);
        return Err(format!(
            "cannot write complete output '{}': {error}",
            path.display()
        ));
    }
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() != 4 {
        return Err(format!(
            "usage: {} MACHINE.bin QUERIES.bin TRANSCRIPT.bin",
            args.first()
                .map(String::as_str)
                .unwrap_or("episode_functor_runtime_rust")
        ));
    }
    if args[1] == args[3] || args[2] == args[3] {
        return Err("output path must differ from input paths".to_owned());
    }
    let maximum_query_size = QUERY_HEADER_SIZE + MAX_QUERIES * QUERY_RECORD_SIZE + HASH_SIZE;
    let machine_bytes = read_bounded(Path::new(&args[1]), MACHINE_SIZE + 1)?;
    let machine = parse_machine(&machine_bytes)?;
    let query_bytes = read_bounded(Path::new(&args[2]), maximum_query_size + 1)?;
    let queries = parse_queries(&query_bytes, &machine)?;
    let transcript = build_transcript(&machine, &queries)?;
    write_new(Path::new(&args[3]), &transcript)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(message) = run(&args) {
        eprintln!("error: {message}");
        let usage_error =
            message.starts_with("usage:") || message == "output path must differ from input paths";
        std::process::exit(if usage_error { 2 } else { 1 });
    }
}
