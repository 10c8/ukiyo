use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use ecow::EcoString;

use super::{Opcode, Value};

#[derive(Debug, Clone)]
struct LineInfo {
    line: usize,
    count: usize,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    code: Vec<u8>,
    lines: Vec<LineInfo>,
    constants: Vec<Arc<Value>>,
}

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            code: Vec::new(),
            lines: Vec::new(),
            constants: Vec::new(),
        }
    }

    pub fn write(&mut self, byte: u8, line: usize) {
        self.code.push(byte);

        if let Some(last) = self.lines.last_mut() {
            if last.line == line {
                last.count += 1;
                return;
            }
        }

        self.lines.push(LineInfo { line, count: 1 });
    }

    pub fn read(&self, pc: usize) -> Option<&u8> {
        self.code.get(pc)
    }

    pub fn get_line(&self, offset: usize) -> Option<usize> {
        let mut total = 0;

        for line in self.lines.iter() {
            total += line.count;

            if total >= offset {
                return Some(line.line);
            }
        }

        None
    }

    pub fn add_constant(&mut self, value: Arc<Value>) -> usize {
        if let Some(idx) = self.find_constant(&value) {
            return idx;
        }

        self.constants.push(value);
        self.constants.len() - 1
    }

    pub fn get_constant(&self, idx: usize) -> Option<&Arc<Value>> {
        self.constants.get(idx)
    }

    fn find_constant(&self, value: &Arc<Value>) -> Option<usize> {
        self.constants.iter().position(|v| v == value)
    }
}

impl Display for Chunk {
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
                        if let Some(value) = self.constants.get(*idx as usize) {
                            data = format!(".. .. {:02x} {:02x}", op, idx).into();

                            format!("LOAD_CONST {:08} = {}", idx, value)
                        } else {
                            panic!("unknown constant");
                        }
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

                    if let Some(value) = self.constants.get(idx) {
                        format!("LOAD_CONST {:08} = {}", idx, value)
                    } else {
                        panic!("unknown constant: {}", idx);
                    }
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
                Opcode::LoadNil => "LOAD_NIL".to_string(),
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
                Opcode::Closure => {
                    let idx = if let Some((idx, _)) = code.next() {
                        idx as usize
                    } else {
                        panic!("missing closure id");
                    };

                    let value = if let Some(value) = self.get_constant(idx) {
                        value.clone()
                    } else {
                        panic!("unknown closure");
                    };

                    if let Value::Closure(closure) = value.as_ref() {
                        data = format!(".. .. .. {:02x}", op).into();

                        let mut line = String::from(format!(
                            "CLOSURE {} * {:03x}",
                            value,
                            closure.upvalues.len()
                        ));

                        for upvalue in &closure.upvalue_refs {
                            line.push_str(&format!(
                                "\n                     |  {} {:03x}",
                                if upvalue.is_local { "LOCAL" } else { "UPVAL" },
                                upvalue.index
                            ));
                        }

                        line
                    } else {
                        panic!("invalid closure");
                    }
                }
                Opcode::Push => "PUSH".to_string(),
                Opcode::Pop => "POP".to_string(),
                Opcode::Negate => "NEG".to_string(),
                Opcode::Add => "ADD".to_string(),
                Opcode::Multiply => "MUL".to_string(),
                Opcode::Divide => "DIV".to_string(),
                Opcode::Subtract => "SUB".to_string(),
                Opcode::Return => "RET".to_string(),
                _ => todo!(),
            };

            let line = self.get_line(i).unwrap();
            let line = if line == last_line {
                "   |".to_string()
            } else {
                last_line = line;
                format!("{:04}", line)
            };

            writeln!(f, "{:#04}  {} {}  {}", i, line, data, instruction)?;
        }

        writeln!(f, "\n[CONSTANTS]")?;
        for (idx, constant) in self.constants.iter().enumerate() {
            writeln!(f, "{:03}  {}", idx, constant)?;
        }

        Ok(())
    }
}
