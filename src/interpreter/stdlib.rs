use std::{borrow::Borrow, sync::Arc};

use ecow::EcoString;
use once_cell::sync::Lazy;
use rand::Rng;
use regex::Regex;
use rustc_hash::FxHashMap;

use crate::{
    llm::{LLMConfig, LLMType, LLM},
    native_arg, native_arg_count, native_fn,
};

use super::{typeof_value, Environment, IResult, InterpreterError, Range, Value};

static IDENT_RE: Lazy<regex::Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap());

pub struct StdLib {}

impl StdLib {
    pub fn load(env: &mut Environment) {
        // Debug functions
        native_fn!("trace"; env: StdLib::std_trace);

        // Logic functions
        native_fn!("eq"; env: StdLib::std_eq);
        native_fn!("neq"; env: StdLib::std_neq);
        native_fn!("gt"; env: StdLib::std_gt);
        native_fn!("lt"; env: StdLib::std_lt);
        native_fn!("gte"; env: StdLib::std_gte);
        native_fn!("lte"; env: StdLib::std_lte);

        // Math functions
        native_fn!("add"; env: StdLib::std_add);
        native_fn!("fadd"; env: StdLib::std_fadd);
        native_fn!("sub"; env: StdLib::std_sub);
        native_fn!("fsub"; env: StdLib::std_fsub);
        native_fn!("mul"; env: StdLib::std_mul);
        native_fn!("fmul"; env: StdLib::std_fmul);
        native_fn!("div"; env: StdLib::std_div);
        native_fn!("fdiv"; env: StdLib::std_fdiv);
        native_fn!("mod"; env: StdLib::std_mod);
        native_fn!("rand"; env: StdLib::std_rand);
        native_fn!("frand"; env: StdLib::std_frand);
        native_fn!("lrand"; env: StdLib::std_lrand);

        // String functions
        native_fn!("lines"; env: StdLib::std_lines);

        // Collection functions
        native_fn!("len"; env: StdLib::std_len);
        native_fn!("join"; env: StdLib::std_join);
        native_fn!("init"; env: StdLib::std_init);
        native_fn!("last"; env: StdLib::std_last);
        native_fn!("head"; env: StdLib::std_head);
        native_fn!("tail"; env: StdLib::std_tail);
        native_fn!("take"; env: StdLib::std_take);
        native_fn!("drop"; env: StdLib::std_drop);
        native_fn!("enumerate"; env: StdLib::std_enumerate);
        native_fn!("map"; env: StdLib::std_map);
        native_fn!("zip"; env: StdLib::std_zip);
        native_fn!("list"; env: StdLib::std_list);
        native_fn!("record"; env: StdLib::std_record);
        native_fn!("filter"; env: StdLib::std_filter);
        native_fn!("has"; env: StdLib::std_has);

        // LLM functions
        native_fn!("gen"; env: StdLib::std_gen);
    }

    // Debug
    pub fn std_trace(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("trace"; args == 1; range);

        let value = native_arg!("trace"; args @ 0; range);

        println!("[trace] {}", value.to_string());

        Ok(Value::Nil.into())
    }

    // Logic
    pub fn std_eq(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("eq"; args == 2; range);

        let a = native_arg!("eq"; args @ 0; range);
        let b = native_arg!("eq"; args @ 1; range);

        Ok(Value::Boolean(a == b).into())
    }

    pub fn std_neq(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("neq"; args == 2; range);

        let a = native_arg!("neq"; args @ 0; range);
        let b = native_arg!("neq"; args @ 1; range);

        Ok(Value::Boolean(a != b).into())
    }

    pub fn std_gt(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("gt"; args == 2; range);

        let a = native_arg!("gt"; args @ 0; range);
        let b = native_arg!("gt"; args @ 1; range);

        match (a.borrow(), b.borrow()) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a > b).into()),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a > b).into()),
            (Value::List(a), Value::List(b)) => Ok(Value::Boolean(a.len() > b.len()).into()),
            (Value::Record(a), Value::Record(b)) => Ok(Value::Boolean(a.len() > b.len()).into()),
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!(
                        "gt: Cannot compare between {} and {}.",
                        typeof_value(&a),
                        typeof_value(&b)
                    )
                    .leak(),
                    range,
                ));
            }
        }
    }

    pub fn std_lt(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("lt"; args == 2; range);

        let a = native_arg!("lt"; args @ 0; range);
        let b = native_arg!("lt"; args @ 1; range);

        match (a.borrow(), b.borrow()) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a < b).into()),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a < b).into()),
            (Value::List(a), Value::List(b)) => Ok(Value::Boolean(a.len() < b.len()).into()),
            (Value::Record(a), Value::Record(b)) => Ok(Value::Boolean(a.len() < b.len()).into()),
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!(
                        "lt: Cannot compare between {} and {}.",
                        typeof_value(&a),
                        typeof_value(&b)
                    )
                    .leak(),
                    range,
                ));
            }
        }
    }

    pub fn std_gte(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("gte"; args == 2; range);

        let a = native_arg!("gte"; args @ 0; range);
        let b = native_arg!("gte"; args @ 1; range);

        match (a.borrow(), b.borrow()) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a >= b).into()),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a >= b).into()),
            (Value::List(a), Value::List(b)) => Ok(Value::Boolean(a.len() >= b.len()).into()),
            (Value::Record(a), Value::Record(b)) => Ok(Value::Boolean(a.len() >= b.len()).into()),
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!(
                        "gte: Cannot compare between {} and {}.",
                        typeof_value(&a),
                        typeof_value(&b)
                    )
                    .leak(),
                    range,
                ));
            }
        }
    }

    pub fn std_lte(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("lte"; args == 2; range);

        let a = native_arg!("lte"; args @ 0; range);
        let b = native_arg!("lte"; args @ 1; range);

        match (a.borrow(), b.borrow()) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a <= b).into()),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a <= b).into()),
            (Value::List(a), Value::List(b)) => Ok(Value::Boolean(a.len() <= b.len()).into()),
            (Value::Record(a), Value::Record(b)) => Ok(Value::Boolean(a.len() <= b.len()).into()),
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!(
                        "lte: Cannot compare between {} and {}.",
                        typeof_value(&a),
                        typeof_value(&b)
                    )
                    .leak(),
                    range,
                ));
            }
        }
    }

    // Math
    pub fn std_add(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("add"; args == 2; range);

        let a = native_arg!("add"; args @ 0 => Number @ range);
        let b = native_arg!("add"; args @ 1 => Number @ range);

        let result = a + b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    pub fn std_fadd(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("fadd"; args == 2; range);

        let a = native_arg!("fadd"; args @ 0 => Number @ range);
        let b = native_arg!("fadd"; args @ 1 => Number @ range);

        let result = a + b;

        Ok(Value::Number(result).into())
    }

    pub fn std_sub(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("sub"; args == 2; range);

        let a = native_arg!("sub"; args @ 0 => Number @ range);
        let b = native_arg!("sub"; args @ 1 => Number @ range);

        let result = a - b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    pub fn std_fsub(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("fsub"; args == 2; range);

        let a = native_arg!("fsub"; args @ 0 => Number @ range);
        let b = native_arg!("fsub"; args @ 1 => Number @ range);

        let result = a - b;

        Ok(Value::Number(result).into())
    }

    pub fn std_mul(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("mul"; args == 2; range);

        let a = native_arg!("mul"; args @ 0 => Number @ range);
        let b = native_arg!("mul"; args @ 1 => Number @ range);

        let result = a * b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    pub fn std_fmul(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("fmul"; args == 2; range);

        let a = native_arg!("fmul"; args @ 0 => Number @ range);
        let b = native_arg!("fmul"; args @ 1 => Number @ range);

        let result = a * b;

        Ok(Value::Number(result).into())
    }

    pub fn std_div(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("div"; args == 2; range);

        let a = native_arg!("div"; args @ 0 => Number @ range);
        let b = native_arg!("div"; args @ 1 => Number @ range);

        let result = a / b;

        Ok(Value::Number((result as i64) as f64).into())
    }

    pub fn std_fdiv(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("fdiv"; args == 2; range);

        let a = native_arg!("fdiv"; args @ 0 => Number @ range);
        let b = native_arg!("fdiv"; args @ 1 => Number @ range);

        let result = a / b;

        Ok(Value::Number(result).into())
    }

    pub fn std_mod(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("mod"; args == 2; range);

        let a = native_arg!("mod"; args @ 0 => Number @ range);
        let b = native_arg!("mod"; args @ 1 => Number @ range);

        Ok(Value::Number(a % b).into())
    }

    pub fn std_rand(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("rand"; args == 2; range);

        let min = native_arg!("rand"; args @ 0 => Number @ range);
        let max = native_arg!("rand"; args @ 1 => Number @ range);

        let mut rng = rand::thread_rng();
        let result = rng.gen_range(*min..=*max) as i64;

        Ok(Value::Number(result as f64).into())
    }

    pub fn std_frand(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("randf"; args == 2; range);

        let min = native_arg!("randf"; args @ 0 => Number @ range);
        let max = native_arg!("randf"; args @ 1 => Number @ range);

        let mut rng = rand::thread_rng();
        let result = rng.gen_range(*min..=*max);

        Ok(Value::Number(result).into())
    }

    pub fn std_lrand(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("randl"; args == 1; range);

        let list = native_arg!("randl"; args @ 0 => List @ range);

        let mut rng = rand::thread_rng();
        let result = rng.gen_range(0..list.len());

        Ok(list[result].clone())
    }

    // Strings
    pub fn std_lines(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("lines"; args == 1; range);

        let value = native_arg!("lines"; args @ 0 => String @ range);

        let mut result = Vec::new();
        for line in value.lines() {
            result.push(Arc::new(Value::String(EcoString::from(line))));
        }

        Ok(Value::List(result.into()).into())
    }

    // Collections
    pub fn std_len(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("len"; args == 1; range);

        let value = args.get(0).unwrap();

        let result = match value.borrow() {
            Value::String(value) => value.len(),
            Value::List(value) => value.len(),
            Value::Record(value) => value.len(),
            _ => Err(InterpreterError::TypeMismatch(
                range,
                "string, list or record",
                typeof_value(value),
            ))?,
        };

        Ok(Value::Number(result as f64).into())
    }

    pub fn std_join(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("join"; args == 2; range);

        let list = native_arg!("join"; args @ 0 => List @ range);
        let separator = native_arg!("join"; args @ 1 => String @ range);

        let mut result = EcoString::new();
        for (i, item) in list.iter().enumerate() {
            if i != 0 {
                result.push_str(separator);
            }

            match item.borrow() {
                Value::String(value) => result.push_str(value),
                _ => result.push_str(&item.to_string()),
            };
        }

        Ok(Value::String(result.into()).into())
    }

    pub fn std_init(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("init"; args == 1; range);

        let mut list = native_arg!("init"; args @ 0 => List @ range).to_owned();

        if list.len() == 0 {
            return Err(InterpreterError::NativeFunctionError(
                "init: List is empty.",
                range,
            ));
        }

        list = list[0..list.len() - 1].to_vec().into();

        Ok(Value::List(list).into())
    }

    pub fn std_last(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("last"; args == 1; range);

        let list = native_arg!("last"; args @ 0 => List @ range);

        let last = match list.last() {
            Some(last) => last,
            None => {
                return Err(InterpreterError::NativeFunctionError(
                    "last: List is empty.",
                    range,
                ))
            }
        };

        Ok(last.clone())
    }

    pub fn std_head(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("head"; args == 1; range);

        let list = native_arg!("head"; args @ 0; range);

        let head = match list.borrow() {
            Value::String(value) => {
                if value.len() == 0 {
                    return Err(InterpreterError::NativeFunctionError(
                        "head: String is empty.",
                        range,
                    ));
                }

                Arc::new(Value::String(EcoString::from(
                    value.chars().next().unwrap(),
                )))
            }
            Value::List(value) => {
                if value.len() == 0 {
                    return Err(InterpreterError::NativeFunctionError(
                        "head: List is empty.",
                        range,
                    ));
                }

                value[0].clone()
            }
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!(
                        "head: Expected string or list, found {}.",
                        typeof_value(list)
                    )
                    .leak(),
                    range,
                ));
            }
        };

        Ok(head)
    }

    pub fn std_tail(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("tail"; args == 1; range);

        let mut list = native_arg!("tail"; args @ 0 => List @ range).to_owned();

        if list.len() == 0 {
            return Err(InterpreterError::NativeFunctionError(
                "tail: List is empty.",
                range,
            ));
        }

        list = list[1..list.len()].to_vec().into();

        Ok(Value::List(list).into())
    }

    pub fn std_take(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("take"; args == 2; range);

        let list = native_arg!("take"; args @ 0 => List @ range);
        let count = native_arg!("take"; args @ 1 => Number @ range);

        let count = *count as usize;
        if count > list.len() {
            return Err(InterpreterError::NativeFunctionError(
                "take: Count is greater than list length.",
                range,
            ));
        }

        let result = list[0..count].to_vec().into();

        Ok(Value::List(result).into())
    }

    pub fn std_drop(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("drop"; args == 2; range);

        let list = native_arg!("drop"; args @ 0 => List @ range);
        let count = native_arg!("drop"; args @ 1 => Number @ range);

        let count = *count as usize;
        if count > list.len() {
            return Err(InterpreterError::NativeFunctionError(
                "drop: Count is greater than list length.",
                range,
            ));
        }

        let result = list[count..list.len()].to_vec().into();

        Ok(Value::List(result).into())
    }

    pub fn std_enumerate(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("enum"; args == 1; range);

        let expr = native_arg!("enum"; args @ 0; range);

        let mut result = Vec::new();

        match expr.borrow() {
            Value::String(value) => {
                for (i, c) in value.chars().enumerate() {
                    result.push(Arc::new(Value::List(
                        vec![
                            Arc::new(Value::Number(i as f64)),
                            Arc::new(Value::String(EcoString::from(c))),
                        ]
                        .into(),
                    )));
                }
            }
            Value::List(value) => {
                for (i, item) in value.iter().enumerate() {
                    result.push(Arc::new(Value::List(
                        vec![Arc::new(Value::Number(i as f64)), item.clone()].into(),
                    )));
                }
            }
            Value::Record(value) => {
                for (i, (key, value)) in value.iter().enumerate() {
                    result.push(Arc::new(Value::List(
                        vec![
                            Arc::new(Value::Number(i as f64)),
                            Arc::new(Value::String(key.clone())),
                            value.clone(),
                        ]
                        .into(),
                    )));
                }
            }
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!("enum: Cannot enumerate a {}.", typeof_value(expr)).leak(),
                    range,
                ));
            }
        }

        Ok(Value::List(result.into()).into())
    }

    pub fn std_map(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("map"; args == 2; range);

        let list = native_arg!("map"; args @ 0; range);
        let closure = native_arg!("map"; args @ 1 => Closure @ range);

        let mut result = Vec::new();

        match list.borrow() {
            Value::List(items) => {
                for item in items.iter() {
                    let value = closure.call(None, vec![item.clone()], None)?;
                    result.push(value);
                }
            }
            Value::Record(items) => {
                for (key, value) in items.iter() {
                    let value = closure.call(
                        None,
                        vec![Arc::new(Value::String(key.clone())), value.clone()].into(),
                        None,
                    )?;
                    result.push(value);
                }
            }
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    format!("map: Cannot map a {}.", typeof_value(list)).leak(),
                    range,
                ));
            }
        };

        Ok(Value::List(result.into()).into())
    }

    pub fn std_list(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("list"; args == 1; range);

        let record = native_arg!("list"; args @ 0 => Record @ range);

        let mut result = Vec::new();
        for (key, value) in record.iter() {
            result.push(Arc::new(Value::List(
                vec![Arc::new(Value::String(key.clone())), value.clone()].into(),
            )));
        }

        Ok(Value::List(result.into()).into())
    }

    pub fn std_record(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("record"; args == 1; range);

        let list = native_arg!("record"; args @ 0 => List @ range);

        let mut result = FxHashMap::default();

        for item in list.iter() {
            match item.borrow() {
                Value::List(pair) => {
                    if pair.len() != 2 {
                        return Err(InterpreterError::NativeFunctionError(
                            "record: List must contain pairs.",
                            range,
                        ));
                    }

                    let key = pair[0].clone();
                    let value = pair[1].clone();

                    match key.borrow() {
                        Value::String(key) => {
                            if !IDENT_RE.is_match(key) {
                                return Err(InterpreterError::NativeFunctionError(
                                    format!("record: \"{}\" is not a valid identifier.", key)
                                        .leak(),
                                    range,
                                ));
                            }

                            result.insert(key.clone(), value);
                        }
                        _ => {
                            return Err(InterpreterError::NativeFunctionError(
                                "record: Key must be a string.",
                                range,
                            ));
                        }
                    }
                }
                _ => {
                    return Err(InterpreterError::NativeFunctionError(
                        "record: List must contain pairs.",
                        range,
                    ));
                }
            }
        }

        Ok(Value::Record(result.into()).into())
    }

    pub fn std_zip(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("zip"; args == 2; range);

        let list_a = native_arg!("zip"; args @ 0 => List @ range);
        let list_b = native_arg!("zip"; args @ 1 => List @ range);

        let mut result = Vec::new();
        for (item_a, item_b) in list_a.iter().zip(list_b.iter()) {
            result.push(Value::List(vec![item_a.clone(), item_b.clone()].into()).into());
        }

        Ok(Value::List(result.into()).into())
    }

    pub fn std_filter(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("filter"; args == 2; range);

        let list = native_arg!("filter"; args @ 0 => List @ range);
        let predicate = native_arg!("filter"; args @ 1 => Closure @ range);

        let mut result = Vec::new();
        for item in list.iter() {
            let value = predicate.call(None, vec![item.clone()], None)?;
            match value.borrow() {
                Value::Boolean(value) => {
                    if *value {
                        result.push(item.clone());
                    }
                }
                _ => {
                    return Err(InterpreterError::TypeMismatch(
                        range,
                        "boolean",
                        typeof_value(&value),
                    ))
                }
            }
        }

        Ok(Value::List(result.into()).into())
    }

    pub fn std_has(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("has"; args == 2; range);

        let collection = native_arg!("has"; args @ 0; range);
        let value = native_arg!("has"; args @ 1; range);

        let result = match collection.borrow() {
            Value::String(collection) => {
                let value = match value.borrow() {
                    Value::String(value) => value,
                    _ => {
                        return Err(InterpreterError::TypeMismatch(
                            range,
                            "string",
                            typeof_value(&value),
                        ))
                    }
                };

                collection.contains(&value.as_str())
            }
            Value::List(collection) => collection.contains(value),
            Value::Record(collection) => {
                let value = match value.borrow() {
                    Value::String(value) => value,
                    _ => {
                        return Err(InterpreterError::TypeMismatch(
                            range,
                            "string",
                            typeof_value(&value),
                        ))
                    }
                };

                collection.contains_key(value)
            }
            _ => {
                return Err(InterpreterError::NativeFunctionError(
                    "has: Value must be a string, list or record.",
                    range,
                ));
            }
        };

        Ok(Value::Boolean(result).into())
    }

    // LLM
    pub fn std_gen(args: Vec<Arc<Value>>, range: Range) -> IResult {
        native_arg_count!("gen"; args == 2; range);

        let prompt = native_arg!("gen"; args @ 0 => String @ range);
        let options = native_arg!("gen"; args @ 1 => Record @ range);

        let context = if let Some(context) = options.get("ctx") {
            match context.borrow() {
                Value::String(value) => Some(value.clone()),
                _ => None,
            }
        } else {
            None
        };

        let max_tokens = if let Some(max_tokens) = options.get("max") {
            match max_tokens.borrow() {
                Value::Number(value) => Some(*value as usize),
                _ => None,
            }
        } else {
            None
        };

        let temperature = if let Some(temperature) = options.get("temp") {
            match temperature.borrow() {
                Value::Number(value) => Some(*value as f32),
                _ => None,
            }
        } else {
            None
        };

        let stop = if let Some(stop) = options.get("stop") {
            match stop.borrow() {
                Value::List(items) => {
                    let mut result = Vec::new();

                    for item in items.iter() {
                        match item.borrow() {
                            Value::String(value) => result.push(value.clone()),
                            _ => {
                                return Err(InterpreterError::NativeFunctionError(
                                    "gen: Stop must be a list of strings.",
                                    range,
                                ));
                            }
                        }
                    }

                    Some(result)
                }
                _ => None,
            }
        } else {
            None
        };

        let config = LLMConfig {
            llm_type: LLMType::OpenAI {
                model: "gpt-3.5-turbo",
                api_key: "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",
                org: "org-UJZWWVz1mXpuSYKHYnchbjHq",
            },
            ..Default::default()
        };
        let llm = LLM::new(config);

        let result = llm.generate(prompt, context, max_tokens, temperature, stop);

        Ok(Value::String(result.into()).into())
    }
}
