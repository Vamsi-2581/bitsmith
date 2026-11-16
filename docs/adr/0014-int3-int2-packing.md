# 0014. INT3 and INT2: asymmetric groups, 24-bit words for INT3

- Status: Accepted
- Date: 2026-11-12

## Context

Below 4 bits, each level counts. INT4 already had a symmetric option, but a symmetric
3-bit grid has 7 usable levels out of 8, and a symmetric 2-bit grid only 3 of 4. Storage is
also less obvious: 3 does not divide 8, so INT3 codes cannot sit in whole bytes or nibbles.

## Decision

- **One integer codec for 4, 3 and 2 bits** (`IntCodec`, with `Int4Codec` kept as the INT4
  name). Group scale in fp16 and zero point in uint8, as for INT4 (ADR 0002).
- **INT3 and INT2 are asymmetric only.** Every code is a level, and the affine range always
  contains zero, so zero weights and padding stay exact.
- **INT3 packs eight codes into one little-endian 24-bit word, stored as three bytes**,
  code `i` at bits `3i` to `3i + 2`. Exactly 3 bits per weight with no padding inside a
  group of eight. Codes may straddle a byte boundary, which costs a shift and an OR to
  unpack but keeps the layout dense. The alternatives were 10 codes in a 32-bit word
  (3.2 bits per weight, 6% waste) or separate 2-bit and 1-bit planes (simpler masks, two
  memory streams).
- **INT2 packs four codes per byte**, the same bit layout as the 2-bit ternary format.
- **Group sizes must be multiples of the packing unit** (8 for INT3, 4 for INT2), so a
  packed group never shares a byte with the next one.
- GPTQ needed no new code: its loop already takes `bits`, and `QLinear.from_qtensor`
  now maps 4, 3 and 2-bit tensors to the matching codec.

## Consequences

- Weight SQNR on Gaussian weights at group 128: INT4 19.9 dB, INT3 13.3 dB, INT2 6.0 dB
  (about 6.6 dB per bit from 4 to 3, as uniform-noise theory predicts; 2 bits falls off
  faster because 4 levels cannot follow the bell shape).
- The fused Triton kernel still handles INT4 only; INT3 and INT2 run on the reference
  path. A 3-bit GEMV would load 3 bytes per 8 codes.
- `make_codec("int3")` used to be the example of an unknown scheme in tests; that is now
  `int5`.
