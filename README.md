# tail

A scripting language for superpowered LLM inference.

## TODOs

### Optimizations

- [ ] Warn about unused functions
  - [ ] Globals
  - [ ] Locals
- [ ] Collapse "load then call" into "load" for constant functions
- [x] Collapse "jump to return" into "return"
  - [ ] Get rid of `NOP`s? This requires that every jump that "passes through" an optimization
        point is correctly offset, so that their target is still correct.
- [ ] Cache function calls for pure functions
- [x] Better string formatting
      Introduce the `BUILD_STR <size: u8>` instruction, which pops `size` strings from the stack
      and concatenates them. This lets us avoid the multiple `CONCAT` instructions that are
      currently generated.
