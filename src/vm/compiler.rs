use std::sync::Arc;

use ecow::EcoString;

use crate::parser::AstNode;

use super::{chunk::Chunk, Closure, Opcode, Value};

#[derive(Debug)]
pub struct Local {
    name: EcoString,
    depth: usize,
}

#[derive(Debug)]
pub enum CompileError {
    Unimplemented,
}

type CResult = Result<(), CompileError>;

pub struct Compiler {
    locals: Vec<Local>,
    scope_depth: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            locals: Vec::new(),
            scope_depth: 0,
        }
    }

    pub fn compile(&mut self, node: &AstNode, chunk: &mut Chunk) -> CResult {
        match node {
            AstNode::Prog { body, .. } => {
                for (i, node) in body.iter().enumerate() {
                    self.compile(node, chunk)?;

                    if i == body.len() - 1 {
                        chunk.pop();
                        chunk.write(Opcode::Return.into(), 1);
                    }
                }

                Ok(())
            }
            AstNode::Block { body, .. } => {
                self.scope_depth += 1;

                for (i, node) in body.iter().enumerate() {
                    self.compile(node, chunk)?;

                    if i == body.len() - 1 {
                        chunk.write(Opcode::Return.into(), 1);
                    } else {
                        chunk.write(Opcode::Pop.into(), 1);
                    }
                }

                self.scope_depth -= 1;

                while !self.locals.is_empty()
                    && self.locals.last().unwrap().depth > self.scope_depth
                {
                    chunk.write(Opcode::Pop.into(), 1);

                    self.locals.pop();
                }

                Ok(())
            }
            AstNode::ExprStmt { expr, .. } => {
                self.compile(expr, chunk)?;

                chunk.write(Opcode::Pop.into(), 1);

                Ok(())
            }
            AstNode::FuncDecl {
                id,
                is_const,
                params,
                body,
                ..
            } => {
                let name = match id.as_ref() {
                    AstNode::Ident { name, .. } => name.clone(),
                    _ => unreachable!(),
                };

                if *is_const {
                    self.compile(body, chunk)?;
                } else {
                    let mut closure_chunk = Chunk::new();
                    self.compile(body, &mut closure_chunk)?;

                    let closure = Closure {
                        name: name.clone(),
                        arity: params.len(),
                        chunk: closure_chunk,
                    };

                    let id = chunk.add_constant(Arc::new(Value::Closure(closure.into())));

                    chunk.write(Opcode::LoadConstant.into(), 1);
                    chunk.write(id as u8, 1);
                }

                if self.scope_depth > 0 {
                    if self.locals.iter().rposition(|l| l.name == name).is_some() {
                        panic!("redeclaration of local variable {}", name);
                    }

                    let local = Local {
                        name,
                        depth: self.scope_depth,
                    };
                    self.locals.push(local);
                } else {
                    let id = chunk.add_constant(Value::String(name).into());

                    chunk.write(Opcode::LoadConstant.into(), 1);
                    chunk.write(id as u8, 1);
                    chunk.write(Opcode::StoreGlobal.into(), 1);
                }

                Ok(())
            }
            AstNode::FuncCall { callee, args, .. } => {
                for arg in args.iter() {
                    self.compile(arg, chunk)?;
                }

                match callee.as_ref() {
                    AstNode::Ident { name, .. } => {
                        let idx = self.locals.iter().rposition(|l| l.name == *name);
                        if let Some(idx) = idx {
                            chunk.write(Opcode::CallLocal.into(), 1);
                            chunk.write(idx as u8, 1);
                        } else {
                            let id = chunk.add_constant(Value::String(name.clone()).into());

                            chunk.write(Opcode::LoadConstant.into(), 1);
                            chunk.write(id as u8, 1);
                            chunk.write(Opcode::CallGlobal.into(), 1);
                        }

                        return Ok(());
                    }
                    _ => {}
                }

                // chunk.write(Opcode::Call.into(), 1);
                // chunk.write(args.len() as u8, 1);

                todo!("compile func call")
            }
            AstNode::Ident { name, .. } => {
                let idx = self.locals.iter().rposition(|l| l.name == *name);
                if let Some(idx) = idx {
                    chunk.write(Opcode::LoadLocal.into(), 1);
                    chunk.write(idx as u8, 1);
                } else {
                    let id = chunk.add_constant(Value::String(name.clone()).into());

                    chunk.write(Opcode::LoadConstant.into(), 1);
                    chunk.write(id as u8, 1);
                    chunk.write(Opcode::LoadGlobal.into(), 1);
                }

                Ok(())
            }
            AstNode::BoolLit { value, .. } => {
                match value {
                    true => chunk.write(Opcode::LoadTrue.into(), 1),
                    false => chunk.write(Opcode::LoadFalse.into(), 1),
                }

                Ok(())
            }
            AstNode::StrLit { value, .. } => {
                let value = Arc::new(Value::String(value.clone()));

                let id = chunk.add_constant(value.clone());

                chunk.write(Opcode::LoadConstant.into(), 1);
                chunk.write(id as u8, 1);

                Ok(())
            }
            AstNode::NumLit { value, .. } => {
                match *value as usize {
                    0 => chunk.write(Opcode::LoadZero.into(), 1),
                    1 => chunk.write(Opcode::LoadOne.into(), 1),
                    _ => {
                        let value = Arc::new(Value::Number(*value));

                        let id = chunk.add_constant(value.clone());

                        chunk.write(Opcode::LoadConstant.into(), 1);
                        chunk.write(id as u8, 1);
                    }
                }

                Ok(())
            }
            _ => todo!("compile {:?}", node),
        }
    }
}
