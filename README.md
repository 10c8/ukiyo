# tail

A scripting language for superpowered LLM inference.

## TODOs

### Optimizations

- [ ] Collapse "load then call" into "load" for constant functions
- [x] Collapse "jump to return" into "return"
  - [ ] Get rid of `NOP`s? This requires that every jump that "passes through" an optimization
        point is correctly offset, so that their target is still correct.
- [ ] Cache function calls for pure functions
