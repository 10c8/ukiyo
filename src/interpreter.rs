use std::{collections::HashMap, fmt};

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
        Config,
    },
};

use crate::{
    lexer::{Lexer, RangeMode},
    parser::{AstNode, CasePatternKind, Parser},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    List(Vec<Value>),
    Record(HashMap<String, Value>),
    Closure(Closure),
    NativeFunction(fn(Vec<Value>, Range) -> IResult),
    Any,
    Nil,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(value) => write!(f, "{}", value),
            Value::Number(value) => write!(f, "{}", value),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Record(record) => {
                write!(f, "{{")?;
                for (i, (key, value)) in record.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} = {}", key, value)?;
                }
                write!(f, "}}")
            }
            Value::Closure(_) => write!(f, "closure"),
            Value::NativeFunction(_) => write!(f, "native function"),
            Value::Any => write!(f, "any"),
            Value::Nil => write!(f, "nil"),
        }
    }
}

fn typeof_value(value: &Value) -> String {
    match value {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::List(_) => "list",
        Value::Record(_) => "record",
        Value::Closure(_) => "closure",
        Value::NativeFunction(_) => "native function",
        Value::Any => "any",
        Value::Nil => "nil",
    }
    .to_string()
}

macro_rules! native_fn {
    ($name:literal; $self:ident: $fn:expr) => {
        $self
            .values
            .insert($name.to_string(), Value::NativeFunction($fn));
    };
}

macro_rules! native_arg_count {
    ($name:literal; $args:ident == $count:expr; $range:expr) => {
        if $args.len() != $count {
            return Err(InterpreterError::NativeFunctionWrongArgumentCount(
                $range,
                $count,
                $args.len(),
            ));
        }
    };
}

macro_rules! native_arg {
    ($name:literal; $args:ident@$index:expr => $type:ident @ $range:expr) => {
        match $args.get($index) {
            Some(Value::$type(value)) => value,
            _ => {
                return Err(InterpreterError::NativeFunctionInvalidArgument(
                    $range,
                    $index,
                    stringify!($type).to_lowercase(),
                    typeof_value($args.get($index).unwrap()),
                ))
            }
        }
    };
}

#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    values: HashMap<String, Value>,
    value_ranges: HashMap<String, Range>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            value_ranges: HashMap::new(),
        }
    }

    pub fn load_stdlib(&mut self) {
        // Math functions
        native_fn!("mod"; self: Self::std_fn_mod);

        // List functions
        native_fn!("len"; self: Self::std_fn_len);
        native_fn!("join"; self: Self::std_fn_join);
        native_fn!("init"; self: Self::std_fn_init);
        native_fn!("last"; self: Self::std_fn_last);
        native_fn!("head"; self: Self::std_fn_head);
        native_fn!("tail"; self: Self::std_fn_tail);
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    pub fn set(&mut self, name: &str, value: Value, range: Range) {
        self.values.insert(name.to_string(), value);
        self.value_ranges.insert(name.to_string(), range);
    }

    pub fn get_range(&self, name: &str) -> Option<&Range> {
        self.value_ranges.get(name)
    }

    fn std_fn_mod(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("mod"; args == 2; range);

        let a = native_arg!("mod"; args @ 0 => Number @ range);
        let b = native_arg!("mod"; args @ 1 => Number @ range);

        Ok(Value::Number(a % b))
    }

    fn std_fn_len(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("len"; args == 1; range);

        let list = native_arg!("len"; args @ 0 => List @ range);

        Ok(Value::Number(list.len() as f64))
    }

    fn std_fn_join(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("join"; args == 2; range);

        let list = native_arg!("join"; args @ 0 => List @ range);
        let separator = native_arg!("join"; args @ 1 => String @ range);

        let mut result = String::new();
        for (i, item) in list.iter().enumerate() {
            if i != 0 {
                result.push_str(separator);
            }

            match item {
                Value::String(value) => result.push_str(value),
                _ => result.push_str(&item.to_string()),
            };
        }

        Ok(Value::String(result))
    }

    fn std_fn_init(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("init"; args == 1; range);

        let mut list = native_arg!("init"; args @ 0 => List @ range).to_owned();

        if list.len() == 0 {
            return Err(InterpreterError::NativeFunctionError(
                "init: list is empty".to_string(),
            ));
        }

        list.pop();

        Ok(Value::List(list))
    }

    fn std_fn_last(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("last"; args == 1; range);

        let list = native_arg!("last"; args @ 0 => List @ range);

        let last = match list.last() {
            Some(last) => last,
            None => {
                return Err(InterpreterError::NativeFunctionError(
                    "last: list is empty".to_string(),
                ))
            }
        };

        Ok(last.to_owned())
    }

    fn std_fn_head(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("head"; args == 1; range);

        let list = native_arg!("head"; args @ 0 => List @ range);

        let head = match list.first() {
            Some(head) => head,
            None => {
                return Err(InterpreterError::NativeFunctionError(
                    "head: list is empty".to_string(),
                ))
            }
        };

        Ok(head.to_owned())
    }

    fn std_fn_tail(args: Vec<Value>, range: Range) -> IResult {
        native_arg_count!("tail"; args == 1; range);

        let mut list = native_arg!("tail"; args @ 0 => List @ range).to_owned();

        if list.len() == 0 {
            return Err(InterpreterError::NativeFunctionError(
                "tail: list is empty".to_string(),
            ));
        }

        list.remove(0);

        Ok(Value::List(list))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    params: Vec<String>,
    body: AstNode,
    env: Environment,
}

type Range = (usize, usize);

#[derive(Debug, Clone)]
pub enum InterpreterError {
    UndefinedVariable(String, Range),
    NotCallable(AstNode, Range),
    WrongArgumentCount(Range, usize, usize),
    PatternTypeMismatch(Range, String, String),
    Redeclaration(String, Option<Range>, Range),
    InvalidFunctionName(Range),
    InvalidFunctionParam(Range),
    NativeFunctionError(String),
    NativeFunctionWrongArgumentCount(Range, usize, usize),
    NativeFunctionInvalidArgument(Range, usize, String, String),
}

type IResult = Result<Value, InterpreterError>;

pub struct Interpreter {
    fmt_range: Option<Range>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self { fmt_range: None }
    }

    pub fn display_error(&mut self, files: SimpleFiles<&str, &str>, error: InterpreterError) {
        let writer = StandardStream::stderr(ColorChoice::Auto);
        let config = Config::default();

        let diagnostic = match error {
            InterpreterError::UndefinedVariable(name, range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("\"{}\" is not defined.", name))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
            InterpreterError::NotCallable(value, range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                let typename = match value {
                    AstNode::Case { .. } => "case expression",
                    AstNode::StrLit { .. } => "string",
                    AstNode::NumLit { .. } => "number",
                    AstNode::List { .. } => "list",
                    AstNode::Record { .. } => "record",
                    _ => unreachable!("not callable: {:#?}", value),
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Cannot call a {}.", typename))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
            }
            InterpreterError::WrongArgumentCount(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Wrong number of arguments.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Expected {} {}, got {}.",
                        expected,
                        if expected == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                        got
                    )])
            }
            InterpreterError::PatternTypeMismatch(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Pattern type mismatch.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Cannot test a {} against a {}.",
                        got, expected
                    )])
            }
            InterpreterError::Redeclaration(name, org_range, new_range) => {
                let new_range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => new_range,
                };

                let mut labels = Vec::new();
                labels.push(Label::primary(0, new_range.0..new_range.1));

                if let Some(org_range) = org_range {
                    labels.push(
                        Label::secondary(0, org_range.0..org_range.1)
                            .with_message("first defined here"),
                    );
                }

                let diagnostic = Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("\"{}\" is already defined.", name))
                    .with_labels(labels);

                if org_range.is_none() {
                    diagnostic
                        .with_notes(vec!["You cannot redefine a built-in function.".to_string()])
                } else {
                    diagnostic
                }
            }
            InterpreterError::InvalidFunctionName(range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Invalid function name.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("A function cannot be called `_`.")])
            }
            InterpreterError::InvalidFunctionParam(range) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Invalid parameter name.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("A parameter cannot be called `_`.")])
            }
            InterpreterError::NativeFunctionError(message) => Diagnostic::error()
                .with_code("run")
                .with_message(message)
                .with_labels(vec![]),
            InterpreterError::NativeFunctionWrongArgumentCount(range, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message("Wrong number of arguments.")
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!(
                        "Expected {} {}, got {}.",
                        expected,
                        if expected == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                        got
                    )])
            }
            InterpreterError::NativeFunctionInvalidArgument(range, index, expected, got) => {
                let range = match self.fmt_range {
                    Some(fmt_range) => fmt_range,
                    None => range,
                };

                Diagnostic::error()
                    .with_code("run")
                    .with_message(format!("Invalid argument at index {}.", index))
                    .with_labels(vec![Label::primary(0, range.0..range.1)])
                    .with_notes(vec![format!("Expected a {}, got a {}.", expected, got)])
            }
        };

        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
    }

    pub fn eval(&mut self, node: &AstNode, environment: &mut Environment) -> IResult {
        let result = match node {
            AstNode::Prog { body, .. } => {
                let mut result = Value::Nil;
                for node in body {
                    result = self.eval(node, environment)?;
                }
                result
            }
            AstNode::ExprStmt { expr, .. } => self.eval(expr, environment)?,
            AstNode::Block { body, .. } => {
                let mut result = Value::Nil;
                for node in body {
                    result = self.eval(node, environment)?;
                }
                result
            }
            AstNode::FuncDecl {
                id: id_node,
                params: fn_params,
                body,
                ..
            } => {
                let id = match id_node {
                    Some(id) => match id.as_ref() {
                        AstNode::Ident { name, .. } => {
                            if name == "_" {
                                return Err(InterpreterError::InvalidFunctionName(id.range()));
                            } else {
                                if environment.get(&name).is_some() {
                                    return Err(InterpreterError::Redeclaration(
                                        name.to_string(),
                                        environment.get_range(&name).copied(),
                                        id.range(),
                                    ));
                                }

                                Some(name.to_owned())
                            }
                        }
                        _ => unreachable!(),
                    },
                    None => None,
                };

                let mut params = Vec::new();
                for param in fn_params {
                    match param {
                        AstNode::Ident { name, .. } => {
                            if name == "_" {
                                return Err(InterpreterError::InvalidFunctionParam(param.range()));
                            } else {
                                params.push(name.to_owned())
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let closure = Value::Closure(Closure {
                    params,
                    body: *body.to_owned(),
                    env: environment.to_owned(),
                });

                if let Some(id) = id {
                    environment.set(&id, closure, id_node.as_ref().unwrap().range());
                }

                Value::Nil
            }
            AstNode::Lambda {
                params: fn_params,
                body,
                ..
            } => {
                let mut params = Vec::new();
                for param in fn_params {
                    match param {
                        AstNode::Ident { name, .. } => {
                            if name == "_" {
                                return Err(InterpreterError::InvalidFunctionParam(param.range()));
                            } else {
                                params.push(name.to_owned())
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let closure = Closure {
                    params,
                    body: *body.to_owned(),
                    env: environment.to_owned(),
                };

                Value::Closure(closure)
            }
            AstNode::Case { expr, cases, .. } => {
                let case_expr = self.eval(expr, environment)?;

                for case in cases {
                    let (pattern, body) = match case {
                        AstNode::CaseBranch { pattern, body, .. } => (pattern, body),
                        _ => unreachable!(),
                    };

                    match **pattern {
                        AstNode::CasePattern {
                            ref kind, ref expr, ..
                        } => match kind {
                            CasePatternKind::Expr => {
                                let expr = self.eval(&expr, environment)?;

                                if match expr {
                                    Value::List(ref list) => {
                                        let expr_items = match case_expr {
                                            Value::List(ref items) => items,
                                            _ => Err(InterpreterError::PatternTypeMismatch(
                                                pattern.range(),
                                                typeof_value(&case_expr),
                                                typeof_value(&expr),
                                            ))?,
                                        };

                                        let mut is_match = true;

                                        if list.len() != expr_items.len() {
                                            is_match = false;
                                        } else {
                                            for (item, expr_item) in
                                                list.iter().zip(expr_items.iter())
                                            {
                                                if *item == Value::Any {
                                                    continue;
                                                }

                                                if item != expr_item {
                                                    is_match = false;
                                                    break;
                                                }
                                            }
                                        }

                                        is_match
                                    }
                                    _ => case_expr == expr,
                                } {
                                    return self.eval(body, environment);
                                }
                            }
                            CasePatternKind::Or => {
                                let list = match self.eval(&expr, environment)? {
                                    Value::List(list) => list,
                                    _ => unreachable!(),
                                };

                                for item in list {
                                    if item == Value::Any || case_expr == item {
                                        return self.eval(body, environment);
                                    }
                                }
                            }
                            CasePatternKind::Any => return self.eval(body, environment),
                        },
                        _ => unreachable!(),
                    }
                }

                Value::Nil
            }
            AstNode::FuncCall {
                callee,
                args: pre_args,
                ..
            } => {
                let mut args = Vec::new();
                for arg in pre_args {
                    args.push(self.eval(arg, environment)?);
                }

                let result = match **callee {
                    AstNode::Ident { ref name, range } => match environment.get(&name) {
                        Some(Value::Closure(closure)) => {
                            self.apply_closure(Some(callee.as_ref()), closure, args)?
                        }
                        Some(Value::NativeFunction(method)) => {
                            self.apply_native(method, args, range)?
                        }
                        Some(value) => {
                            if args.len() != 0 {
                                return Err(InterpreterError::NotCallable(
                                    *callee.to_owned(),
                                    callee.range(),
                                ));
                            } else {
                                value.to_owned()
                            }
                        }
                        _ => {
                            return Err(InterpreterError::UndefinedVariable(
                                name.to_string(),
                                range,
                            ))
                        }
                    },
                    AstNode::FuncCall { .. } => {
                        let callee_closure = self.eval(callee.as_ref(), environment)?;
                        let callee_closure = match callee_closure {
                            Value::Closure(closure) => closure,
                            _ => unreachable!(),
                        };

                        self.apply_closure(Some(callee.as_ref()), &callee_closure, args)?
                    }
                    _ => {
                        return Err(InterpreterError::NotCallable(
                            *callee.to_owned(),
                            callee.range(),
                        ));
                    }
                };

                result
            }
            AstNode::IterationOp { id, expr, body, .. } => {
                let id = match id.as_ref() {
                    AstNode::Ident { ref name, .. } => name,
                    _ => unreachable!(),
                };

                let expr = self.eval(expr, environment)?;
                match expr {
                    Value::List(items) => {
                        let mut result = Vec::new();

                        let env = environment.to_owned();
                        let closure = Closure {
                            params: vec![id.to_owned()],
                            body: *body.to_owned(),
                            env,
                        };

                        for item in items {
                            result.push(self.apply_closure(None, &closure, vec![item])?);
                        }

                        Value::List(result)
                    }
                    _ => panic!("{:#?} is not iterable", expr),
                }
            }
            AstNode::Ident { ref name, range } => {
                if *name == "_".to_string() {
                    return Ok(Value::Any);
                }

                match environment.get(&name) {
                    Some(Value::Closure(closure)) => {
                        self.apply_closure(Some(&node), closure, vec![])?
                    }
                    Some(Value::NativeFunction(method)) => {
                        self.apply_native(method, vec![], *range)?
                    }
                    Some(value) => value.to_owned(),
                    None => {
                        return Err(InterpreterError::UndefinedVariable(
                            name.to_string(),
                            node.range(),
                        ))
                    }
                }
            }
            AstNode::StrLit {
                value,
                is_format,
                range,
            } => {
                if *is_format {
                    self.eval_fmt_string(&value, *range, environment)?
                } else {
                    Value::String(value.to_owned())
                }
            }
            AstNode::NumLit { value, .. } => Value::Number(*value),
            AstNode::List { items, .. } => {
                let mut result = Vec::new();

                for item in items {
                    let value = self.eval(item, environment)?;
                    result.push(value);
                }

                Value::List(result)
            }
            AstNode::Record { keys, values, .. } => {
                let mut record = HashMap::new();

                for (key, value) in keys.iter().zip(values.iter()) {
                    let key = match key {
                        AstNode::StrLit { value, .. } => value.to_owned(),
                        _ => unreachable!(),
                    };
                    let value = self.eval(value, environment)?;

                    record.insert(key, value);
                }

                Value::Record(record)
            }
            AstNode::Range {
                mode, start, end, ..
            } => {
                let start = match self.eval(start, environment)? {
                    Value::Number(value) => value,
                    _ => panic!("start is not a number"),
                };

                let end = match self.eval(end, environment)? {
                    Value::Number(value) => value,
                    _ => panic!("end is not a number"),
                };

                let mut result = Vec::new();

                match mode {
                    RangeMode::Inclusive => {
                        for i in start as usize..=end as usize {
                            result.push(Value::Number(i as f64));
                        }
                    }
                    RangeMode::Exclusive => {
                        for i in start as usize..end as usize {
                            result.push(Value::Number(i as f64));
                        }
                    }
                }

                Value::List(result)
            }
            _ => todo!("eval {:?}", node),
        };

        Ok(result)
    }

    fn apply_closure(
        &mut self,
        callee: Option<&AstNode>,
        closure: &Closure,
        args: Vec<Value>,
    ) -> IResult {
        if closure.params.len() != args.len() {
            let callee = match callee {
                Some(callee) => callee,
                None => unreachable!(),
            };

            return Err(InterpreterError::WrongArgumentCount(
                callee.range(),
                closure.params.len(),
                args.len(),
            ));
        }

        let mut env = closure.env.to_owned();
        for (param, arg) in closure.params.iter().zip(args.iter()) {
            env.set(param, arg.to_owned(), (0, 0));
        }

        self.eval(&closure.body, &mut env)
    }

    fn apply_native(
        &mut self,
        method: &fn(Vec<Value>, Range) -> IResult,
        args: Vec<Value>,
        range: Range,
    ) -> IResult {
        method(args, range)
    }

    fn eval_fmt_string(
        &mut self,
        value: &String,
        range: Range,
        environment: &mut Environment,
    ) -> IResult {
        let mut result = String::new();
        let mut chars = value.chars().peekable();

        let mut is_inside = false;
        let mut buffer = String::new();

        while let Some(c) = chars.next() {
            if c == '{' {
                is_inside = true;
                continue;
            }

            if c == '}' {
                is_inside = false;

                let mut lexer = Lexer::new(&buffer);
                lexer.lex().expect("fmt: lexer error");

                let mut parser = Parser::new(lexer);
                let ast = parser.parse().expect("fmt: parser error");

                self.fmt_range = Some(range);

                let value = self.eval(&ast, environment)?;
                let value = value.to_string();

                self.fmt_range = None;

                result.push_str(&value);

                buffer.clear();
                continue;
            }

            if is_inside {
                buffer.push(c);
                continue;
            }

            result.push(c);
        }

        Ok(Value::String(result))
    }
}
