use std::fmt::{Display, Formatter};

use ecow::EcoString;

use super::Opcode;

#[derive(Debug, Clone)]
struct LineInfo {
    line: usize,
    count: usize,
}

#[derive(Debug, Clone)]
pub struct Block {
    code: Vec<u8>,
    lines: Vec<LineInfo>,
}

impl Block {
    pub fn new() -> Block {
        Block {
            code: Vec::new(),
            lines: Vec::new(),
        }
    }

    pub fn code(&self) -> &[u8] {
        &self.code
    }

    pub fn size(&self) -> usize {
        self.code.len()
    }

    pub fn insert(&mut self, byte: u8, line: usize) {
        self.code.push(byte);

        if let Some(last) = self.lines.iter_mut().rfind(|l| l.line == line) {
            last.count += 1;
            return;
        }

        self.lines.push(LineInfo { line, count: 1 });
    }

    pub fn set(&mut self, offset: usize, byte: u8) {
        self.code[offset] = byte;
    }

    pub fn read(&self, pc: usize) -> Option<&u8> {
        self.code.get(pc)
    }

    pub fn get_line(&self, offset: usize) -> Option<usize> {
        let mut total = 0;

        for line in self.lines.iter() {
            total += line.count;

            if total > offset {
                return Some(line.line);
            }
        }

        None
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut code = self.code.iter().enumerate();

        let mut data: EcoString;
        let mut last_line = 0;

        writeln!(f, "[CODE]")?;
        while let Some((i, op)) = code.next() {
            data = format!(".. .. .. {:02x}", op).into();

            let instruction = match Opcode::from(*op) {
                Opcode::LoadConstant => {
                    if let Some((_, idx)) = code.next() {
                        data = format!(".. .. {:02x} {:02x}", op, idx).into();

                        format!("LD_CONST {:06x}", idx)
                    } else {
                        panic!("missing constant idx");
                    }
                }
                Opcode::LoadConstantLong => {
                    let idx_lo = code.next();
                    let idx_md = code.next();
                    let idx_hi = code.next();

                    let idx = if let (Some((_, idx_lo)), Some((_, idx_md)), Some((_, idx_hi))) =
                        (idx_lo, idx_md, idx_hi)
                    {
                        data = format!("{:02x} {:02x} {:02x} {:02x}", op, idx_lo, idx_md, idx_hi)
                            .into();

                        (*idx_lo as usize) | ((*idx_md as usize) << 8) | ((*idx_hi as usize) << 16)
                    } else {
                        panic!("missing constant idx");
                    };

                    format!("LD_CONST {:06x}", idx)
                }
                Opcode::StoreGlobal => "ST_GLOBAL".to_string(),
                Opcode::LoadLocal => {
                    if let Some((_, idx)) = code.next() {
                        data = format!(".. .. {:02x} {:02x}", op, idx).into();

                        format!("LD_LOCAL {}", idx)
                    } else {
                        panic!("missing local idx");
                    }
                }
                Opcode::LoadGlobal => "LD_GLOBAL".to_string(),
                Opcode::LoadUpvalue => {
                    if let Some((_, idx)) = code.next() {
                        data = format!(".. .. {:02x} {:02x}", op, idx).into();

                        format!("LD_UPVAL {}", idx)
                    } else {
                        panic!("missing upvalue idx");
                    }
                }
                Opcode::LoadZero => "LOAD_ZERO".to_string(),
                Opcode::LoadOne => "LOAD_ONE".to_string(),
                Opcode::LoadTwo => "LOAD_TWO".to_string(),
                Opcode::LoadMinusOne => "LOAD_MINUS_ONE".to_string(),
                Opcode::LoadTrue => "LOAD_TRUE".to_string(),
                Opcode::LoadFalse => "LOAD_FALSE".to_string(),
                Opcode::Call => {
                    if let Some((_, argc)) = code.next() {
                        data = format!(".. .. {:02x} {:02x}", op, argc).into();

                        format!("CALL {}", argc)
                    } else {
                        panic!("missing argument count");
                    }
                }
                Opcode::LoadClosure => {
                    let idx = if let Some((_, idx)) = code.next() {
                        *idx as usize
                    } else {
                        panic!("missing closure id");
                    };

                    data = format!(".. .. {:02x} {:02x}", op, idx).into();

                    format!("LD_CLOSURE {:02x}", idx)
                }
                Opcode::LoadList => {
                    let size_lo = code.next();
                    let size_hi = code.next();

                    let size = if let (Some((_, size_lo)), Some((_, size_hi))) = (size_lo, size_hi)
                    {
                        data = format!(".. {:02x} {:02x} {:02x}", op, size_lo, size_hi).into();

                        (*size_lo as usize) | ((*size_hi as usize) << 8)
                    } else {
                        panic!("missing list size");
                    };

                    format!("LD_LIST {}", size)
                }
                Opcode::LoadIterator => "LD_ITER".to_string(),
                Opcode::LoopIterator => {
                    let end_lo = code.next();
                    let end_md = code.next();
                    let end_hi = code.next();

                    let end = if let (Some((_, end_lo)), Some((_, end_md)), Some((_, end_hi))) =
                        (end_lo, end_md, end_hi)
                    {
                        data = format!("{:02x} {:02x} {:02x} {:02x}", op, end_lo, end_md, end_hi)
                            .into();

                        (*end_lo as usize) | ((*end_md as usize) << 8) | ((*end_hi as usize) << 16)
                    } else {
                        panic!("missing end offset");
                    };

                    format!("LOOP_ITER {:06x} ({:04x})", end, i + end as usize)
                }
                Opcode::IteratorAppend => {
                    let back_lo = code.next();
                    let back_md = code.next();
                    let back_hi = code.next();

                    let back = if let (Some((_, back_lo)), Some((_, back_md)), Some((_, back_hi))) =
                        (back_lo, back_md, back_hi)
                    {
                        data =
                            format!("{:02x} {:02x} {:02x} {:02x}", op, back_lo, back_md, back_hi)
                                .into();

                        (*back_lo as usize)
                            | ((*back_md as usize) << 8)
                            | ((*back_hi as usize) << 16)
                    } else {
                        panic!("missing back offset");
                    };

                    format!("ITER_APPEND {:06x} ({:04x})", back, i - back as usize)
                }
                Opcode::BuildString => {
                    let size = code.next();

                    if let Some((_, size)) = size {
                        data = format!(".. .. {:02x} {:02x}", op, size).into();

                        format!("BUILD_STR {}", size)
                    } else {
                        panic!("missing string size");
                    }
                }
                Opcode::LoadRangeInclusive => "LD_RANGE_INC".to_string(),
                Opcode::LoadRangeExclusive => "LD_RANGE_EXC".to_string(),
                Opcode::Pop => "POP".to_string(),
                Opcode::PopTop => "POP_TOP".to_string(),
                Opcode::Jump => {
                    let offset_lo = code.next();
                    let offset_md = code.next();
                    let offset_hi = code.next();

                    let offset =
                        if let (Some((_, offset_lo)), Some((_, offset_mi)), Some((_, offset_hi))) =
                            (offset_lo, offset_md, offset_hi)
                        {
                            data = format!(
                                "{:02x} {:02x} {:02x} {:02x}",
                                op, offset_lo, offset_mi, offset_hi
                            )
                            .into();

                            (*offset_lo as usize)
                                | ((*offset_mi as usize) << 8)
                                | ((*offset_hi as usize) << 16)
                        } else {
                            panic!("missing jump offset");
                        };

                    format!("JMP {:06x} ({:04x})", offset, i + offset as usize)
                }
                Opcode::JumpIfFalse => {
                    let offset_lo = code.next();
                    let offset_md = code.next();
                    let offset_hi = code.next();

                    let offset =
                        if let (Some((_, offset_lo)), Some((_, offset_mi)), Some((_, offset_hi))) =
                            (offset_lo, offset_md, offset_hi)
                        {
                            data = format!(
                                "{:02x} {:02x} {:02x} {:02x}",
                                op, offset_lo, offset_mi, offset_hi
                            )
                            .into();

                            (*offset_lo as usize)
                                | ((*offset_mi as usize) << 8)
                                | ((*offset_hi as usize) << 16)
                        } else {
                            panic!("missing jump offset");
                        };

                    format!("JF {:06x} ({:04x})", offset, i + offset as usize)
                }
                Opcode::JumpBack => {
                    let offset_lo = code.next();
                    let offset_md = code.next();
                    let offset_hi = code.next();

                    let offset =
                        if let (Some((_, offset_lo)), Some((_, offset_mi)), Some((_, offset_hi))) =
                            (offset_lo, offset_md, offset_hi)
                        {
                            data = format!(
                                "{:02x} {:02x} {:02x} {:02x}",
                                op, offset_lo, offset_mi, offset_hi
                            )
                            .into();

                            (*offset_lo as usize)
                                | ((*offset_mi as usize) << 8)
                                | ((*offset_hi as usize) << 16)
                        } else {
                            panic!("missing jump offset");
                        };

                    format!("JB {:06x} ({:04x})", offset, i as isize - offset as isize)
                }
                Opcode::Equals => "EQ".to_string(),
                Opcode::NotEquals => "NEQ".to_string(),
                Opcode::GreaterThan => "GT".to_string(),
                Opcode::LessThan => "LT".to_string(),
                Opcode::GreaterThanOrEqual => "GTE".to_string(),
                Opcode::LessThanOrEqual => "LTE".to_string(),
                Opcode::Negate => "NEG".to_string(),
                Opcode::Add => "ADD".to_string(),
                Opcode::Multiply => "MUL".to_string(),
                Opcode::Divide => "DIV".to_string(),
                Opcode::Subtract => "SUB".to_string(),
                Opcode::Concatenate => "CONCAT".to_string(),
                Opcode::Return => "RET".to_string(),
                Opcode::Nop => "NOP".to_string(),
                _ => format!("opcode: {:?}", Opcode::from(*op)),
            };

            let line = self.get_line(i).unwrap();
            let line = if line == last_line {
                "    ".to_string()
            } else {
                last_line = line;
                format!("{:>4}", line)
            };

            writeln!(f, "{:04x}  {} | {}  {}", i, line, data, instruction)?;
        }

        // writeln!(f, "\n[CONSTANTS]")?;
        // for (idx, constant) in self.constants.iter().enumerate() {
        //     writeln!(f, "{:03}  {:?}", idx, constant)?;
        // }

        Ok(())
    }
}
