pub mod chunk;
pub mod compiler;

use std::{
    fmt::{Display, Formatter},
    sync::{Arc, Mutex},
};

use chunk::Chunk;
use ecow::EcoString;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use serde::de::value;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    LoadConstant,
    LoadConstantLong,
    StoreGlobal,
    LoadLocal,
    LoadGlobal,
    LoadNil,
    LoadTrue,
    LoadFalse,
    LoadZero,
    LoadOne,
    LoadMinusOne,
    CallLocal,
    CallGlobal,
    Push,
    Pop,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Return,
    LAST,
}

impl From<u8> for Opcode {
    fn from(byte: u8) -> Self {
        assert!(byte <= Opcode::LAST as u8);
        unsafe { std::mem::transmute(byte) }
    }
}

impl Into<u8> for Opcode {
    fn into(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone)]
pub struct Closure {
    name: EcoString,
    arity: usize,
    chunk: Chunk,
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.arity == other.arity
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    String(EcoString),
    Number(f64),
    Closure(Arc<Closure>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Number(n) => write!(f, "{:?}", n),
            Value::Closure(c) => write!(f, "<fn {}:{}>", c.name, c.arity),
        }
    }
}

pub struct Stack {
    stack: Vec<Arc<Value>>,
}

impl Stack {
    pub fn new() -> Stack {
        Stack { stack: Vec::new() }
    }

    pub fn push(&mut self, value: Arc<Value>) {
        self.stack.push(value);
    }

    pub fn pop(&mut self) -> Option<Arc<Value>> {
        self.stack.pop()
    }

    pub fn peek(&self, n: usize) -> Option<&Arc<Value>> {
        self.stack.get(n)
    }
}

impl Display for Stack {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[STACK]\n")?;

        for value in self.stack.iter() {
            write!(f, "{}\n", value)?;
        }

        Ok(())
    }
}

pub struct CallFrame {
    closure: Arc<Mutex<Closure>>,
    pc: usize,
    base: usize,
}

impl CallFrame {
    pub fn get_closure(&self) -> &Arc<Mutex<Closure>> {
        &self.closure
    }
}

#[derive(Debug)]
pub enum VMError {
    RuntimeError,
}

type VMResult = Result<(), VMError>;

static NIL_VALUE: Lazy<Arc<Value>> = Lazy::new(|| Arc::new(Value::Nil));
static TRUE_VALUE: Lazy<Arc<Value>> = Lazy::new(|| Arc::new(Value::Bool(true)));
static FALSE_VALUE: Lazy<Arc<Value>> = Lazy::new(|| Arc::new(Value::Bool(false)));
static ZERO_VALUE: Lazy<Arc<Value>> = Lazy::new(|| Arc::new(Value::Number(0.0)));
static ONE_VALUE: Lazy<Arc<Value>> = Lazy::new(|| Arc::new(Value::Number(1.0)));
static MINUS_ONE_VALUE: Lazy<Arc<Value>> = Lazy::new(|| Arc::new(Value::Number(-1.0)));

pub struct VM {
    frames: Vec<CallFrame>,
    pub stack: Stack,
    globals: FxHashMap<EcoString, Arc<Value>>,
}

impl VM {
    pub fn new() -> VM {
        VM {
            frames: Vec::new(),
            stack: Stack::new(),
            globals: FxHashMap::default(),
        }
    }

    pub fn interpret(&mut self, chunk: Chunk) -> VMResult {
        let frame = CallFrame {
            closure: Arc::new(Mutex::new(Closure {
                name: "prog".into(),
                arity: 0,
                chunk,
            })),
            pc: 0,
            base: 0,
        };
        self.frames.push(frame);

        self.run()
    }

    fn run(&mut self) -> VMResult {
        loop {
            if let Some(op) = self.read_byte() {
                match op {
                    Opcode::LoadConstant => self.op_ld_const(false)?,
                    Opcode::LoadConstantLong => self.op_ld_const(true)?,
                    Opcode::StoreGlobal => self.op_st_global()?,
                    Opcode::LoadLocal => self.op_ld_local()?,
                    Opcode::LoadGlobal => self.op_ld_global()?,
                    Opcode::LoadNil => self.stack.push(NIL_VALUE.clone()),
                    Opcode::LoadTrue => self.stack.push(TRUE_VALUE.clone()),
                    Opcode::LoadFalse => self.stack.push(FALSE_VALUE.clone()),
                    Opcode::LoadZero => self.stack.push(ZERO_VALUE.clone()),
                    Opcode::LoadOne => self.stack.push(ONE_VALUE.clone()),
                    Opcode::LoadMinusOne => self.stack.push(MINUS_ONE_VALUE.clone()),
                    Opcode::CallLocal => self.op_call_local()?,
                    Opcode::CallGlobal => self.op_call_global()?,
                    Opcode::Push => self.op_push()?,
                    Opcode::Pop => self.op_pop()?,
                    Opcode::Negate => self.op_neg()?,
                    Opcode::Add => self.op_add()?,
                    Opcode::Subtract => self.op_sub()?,
                    Opcode::Multiply => self.op_mul()?,
                    Opcode::Divide => self.op_div()?,
                    Opcode::Return => {
                        self.frames.pop();

                        if self.frames.is_empty() {
                            break;
                        }
                    }
                    _ => todo!("unknown opcode: {:?}", op),
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    fn read_byte(&mut self) -> Option<Opcode> {
        let frame = self.frames.last_mut().unwrap();

        let op = {
            let chunk = &frame.get_closure().lock().unwrap().chunk;

            if let Some(op) = chunk.read(frame.pc) {
                Some((*op).into())
            } else {
                None
            }
        };

        if op.is_some() {
            frame.pc += 1;
        }

        op
    }

    // LD_CONST <id: u8>
    fn op_ld_const(&mut self, is_long: bool) -> VMResult {
        let id = if is_long {
            let id_lo = self.read_byte();
            let id_md = self.read_byte();
            let id_hi = self.read_byte();

            if let (Some(id_lo), Some(id_md), Some(id_hi)) = (id_lo, id_md, id_hi) {
                (id_lo as usize) | ((id_md as usize) << 8) | ((id_hi as usize) << 16)
            } else {
                println!("ld_const: missing constant id");
                return Err(VMError::RuntimeError);
            }
        } else {
            if let Some(id) = self.read_byte() {
                id as usize
            } else {
                println!("ld_const: missing constant id");
                return Err(VMError::RuntimeError);
            }
        };

        let frame = self.frames.last_mut().unwrap();
        let chunk = &frame.get_closure().lock().unwrap().chunk;

        if let Some(constant) = chunk.get_constant(id as usize) {
            self.stack.push(constant.clone());
            return Ok(());
        }

        println!("ld_const: unknown constant");
        Err(VMError::RuntimeError)
    }

    // ST_GLOBAL
    fn op_st_global(&mut self) -> VMResult {
        if let Some(name) = self.stack.pop() {
            match name.as_ref() {
                Value::String(name) => {
                    if let Some(value) = self.stack.pop() {
                        if self.globals.get(name).is_some() {
                            println!("st_global: redefinition of variable");
                            return Err(VMError::RuntimeError);
                        }

                        self.globals.insert(name.clone(), value.clone());
                        return Ok(());
                    } else {
                        println!("st_global: missing value");
                        return Err(VMError::RuntimeError);
                    }
                }
                _ => {
                    println!("st_global: variable name must be a string");
                    return Err(VMError::RuntimeError);
                }
            }
        }

        println!("st_global: invalid variable name");
        Err(VMError::RuntimeError)
    }

    // LD_LOCAL <idx: u8>
    fn op_ld_local(&mut self) -> VMResult {
        if let Some(idx) = self.read_byte() {
            if let Some(value) = self.stack.peek(idx as usize) {
                self.stack.push(value.clone());
                return Ok(());
            } else {
                println!("ld_local: undefined variable");
                return Err(VMError::RuntimeError);
            }
        }

        println!("ld_local: unknown variable");
        Err(VMError::RuntimeError)
    }

    // LD_GLOBAL
    fn op_ld_global(&mut self) -> VMResult {
        if let Some(name) = self.stack.pop() {
            match name.as_ref() {
                Value::String(name) => {
                    if let Some(value) = self.globals.get(name) {
                        self.stack.push(value.clone());
                        return Ok(());
                    } else {
                        println!("ld_global: undefined variable '{}'", name);
                        return Err(VMError::RuntimeError);
                    }
                }
                _ => {
                    println!("ld_global: variable name must be a string");
                    return Err(VMError::RuntimeError);
                }
            }
        }

        println!("ld_global invalid variable name");
        Err(VMError::RuntimeError)
    }

    // CALL_LOCAL
    fn op_call_local(&mut self) -> VMResult {
        if let Some(id) = self.read_byte() {
            if let Some(closure) = self.stack.peek(id as usize) {
                if let Value::Closure(closure) = closure.as_ref() {
                    let frame = CallFrame {
                        closure: Arc::new(Mutex::new(closure.as_ref().clone())),
                        pc: 0,
                        base: self.stack.stack.len() - (id as usize) - 1,
                    };
                    self.frames.push(frame);

                    return Ok(());
                } else {
                    println!("call_local: value is not a function");
                    return Err(VMError::RuntimeError);
                }
            } else {
                println!("call_local: undefined function");
                return Err(VMError::RuntimeError);
            }
        }

        println!("call_local: unknown function");
        Err(VMError::RuntimeError)
    }

    // CALL_GLOBAL
    fn op_call_global(&mut self) -> VMResult {
        if let Some(name) = self.stack.pop() {
            match name.as_ref() {
                Value::String(name) => {
                    if name == "print" {
                        if let Some(name) = self.stack.pop() {
                            println!(">>> CALL print {}", name);
                            self.stack.push(NIL_VALUE.clone());

                            return Ok(());
                        } else {
                            println!("call_global: missing argument");
                            return Err(VMError::RuntimeError);
                        }
                    } else {
                        if let Some(value) = self.globals.get(name) {
                            match value.as_ref() {
                                Value::Closure(closure) => {
                                    let frame = CallFrame {
                                        closure: Arc::new(Mutex::new(closure.as_ref().clone())),
                                        pc: 0,
                                        base: self.stack.stack.len(),
                                    };
                                    self.frames.push(frame);

                                    return Ok(());
                                }
                                _ => {
                                    self.stack.push(value.clone());
                                    return Ok(());
                                }
                            }
                        } else {
                            println!("call_global: undefined function");
                            return Err(VMError::RuntimeError);
                        }
                    }
                }
                _ => panic!("call_global: function name must be a string"),
            }
        }

        println!("call_global: invalid function name");
        Err(VMError::RuntimeError)
    }

    // PUSH
    fn op_push(&mut self) -> VMResult {
        todo!()
    }

    // POP
    fn op_pop(&mut self) -> VMResult {
        if self.stack.pop().is_some() {
            return Ok(());
        }

        println!("pop: stack underflow");
        Err(VMError::RuntimeError)
    }

    // NEG
    fn op_neg(&mut self) -> VMResult {
        if let Some(value) = self.stack.pop() {
            if let Value::Number(n) = value.as_ref() {
                self.stack.push(Value::Number(-n).into());
                return Ok(());
            }
        }

        println!("neg: invalid operand");
        Err(VMError::RuntimeError)
    }

    // ADD
    fn op_add(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a + b).into());
                return Ok(());
            }
        }

        println!("add: invalid operands");
        Err(VMError::RuntimeError)
    }

    // SUB
    fn op_sub(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a - b).into());
                return Ok(());
            }
        }

        println!("sub: invalid operands");
        Err(VMError::RuntimeError)
    }

    // MUL
    fn op_mul(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a * b).into());
                return Ok(());
            }
        }

        println!("mul: invalid operands");
        Err(VMError::RuntimeError)
    }

    // DIV
    fn op_div(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a / b).into());
                return Ok(());
            }
        }

        println!("div: invalid operands");
        Err(VMError::RuntimeError)
    }
}
