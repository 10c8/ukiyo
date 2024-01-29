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

    pub fn pop(&mut self) {
        self.code.pop();
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
        if let Some(id) = self.find_constant(&value) {
            return id;
        }

        self.constants.push(value);
        self.constants.len() - 1
    }

    pub fn get_constant(&self, id: usize) -> Option<&Arc<Value>> {
        self.constants.get(id)
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
            data = format!(".. .. .. {:#02}", op).into();

            let instruction = match Opcode::from(*op) {
                Opcode::LoadConstant => {
                    if let Some((_, id)) = code.next() {
                        if let Some(value) = self.constants.get(*id as usize) {
                            data = format!(".. .. {:#02} {:#02}", op, id).into();

                            format!("LOAD_CONST {:08} = {}", id, value)
                        } else {
                            panic!("unknown constant");
                        }
                    } else {
                        panic!("missing constant id");
                    }
                }
                Opcode::LoadConstantLong => {
                    let id_lo = code.next();
                    let id_md = code.next();
                    let id_hi = code.next();

                    let id = if let (Some((_, id_lo)), Some((_, id_md)), Some((_, id_hi))) =
                        (id_lo, id_md, id_hi)
                    {
                        data =
                            format!("{:#02} {:#02} {:#02} {:#02}", op, id_lo, id_md, id_hi).into();

                        (*id_lo as usize) | ((*id_md as usize) << 8) | ((*id_hi as usize) << 16)
                    } else {
                        panic!("missing constant id");
                    };

                    if let Some(value) = self.constants.get(id) {
                        format!("LOAD_CONST {:08} = {}", id, value)
                    } else {
                        panic!("unknown constant");
                    }
                }
                Opcode::StoreGlobal => "ST_GLOBAL".to_string(),
                Opcode::LoadLocal => {
                    if let Some((_, id)) = code.next() {
                        data = format!(".. .. {:#02} {:#02}", op, id).into();

                        format!("LD_LOCAL {}", id)
                    } else {
                        panic!("missing local id");
                    }
                }
                Opcode::LoadGlobal => "LD_GLOBAL".to_string(),
                Opcode::LoadZero => "LOAD_ZERO".to_string(),
                Opcode::LoadOne => "LOAD_ONE".to_string(),
                Opcode::LoadMinusOne => "LOAD_MINUS_ONE".to_string(),
                Opcode::LoadNil => "LOAD_NIL".to_string(),
                Opcode::LoadTrue => "LOAD_TRUE".to_string(),
                Opcode::LoadFalse => "LOAD_FALSE".to_string(),
                Opcode::CallLocal => {
                    if let Some((_, id)) = code.next() {
                        data = format!(".. .. {:#02} {:#02}", op, id).into();

                        format!("CALL_LOCAL {}", id)
                    } else {
                        panic!("missing local id");
                    }
                }
                Opcode::CallGlobal => "CALL_GLOBAL".to_string(),
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
        for (id, constant) in self.constants.iter().enumerate() {
            writeln!(f, "{:03}  {}", id, constant)?;
        }

        Ok(())
    }
}
