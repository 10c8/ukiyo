mod interpreter;
mod lexer;
mod parser;
mod scanner;

use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use interpreter::Interpreter;

use crate::lexer::ToDiagnostic;

fn main() {
    /*
        let example = r#"
    fruits -> ["apples", "bananas", "oranges", "limes"]
    flavors -> {
        "fruit"     = ["sweet", "sour", "bitter"],
        "vegetable" = ["bitter", "salty", "umami"],
        "meat"      = ["salty", "umami", "sweet"],
    }

    get_color fruit ->
        color -> case fruit of
            "apples"            => "red"
            "bananas" | "limes" => "yellow"
            "oranges"           => "orange"
            _                   => gen %% { "ctx" = ["give me the fruit color"], "temp" = 0.3 }

        $"{fruit} are {color}"

    fruit_colors ->
        case (len fruits) of
            1 => get_color (head fruits)
            _ =>
                start -> join ((init fruits) |f| get_color f) ", "
                end   -> get_color (last fruits)

                $"{start} and {end}"
    "#;
    */

    let example = r#"
fruits -> ["apples", "bananas", "oranges", "limes", "avocado"]

fruits |f|
  color -> case f of
    "apples"            => "red"
    "oranges"           => "orange"
    "bananas" | "limes" => "yellow"
    _                   => "some color"

  $"{fuck} are {color}"
"#;

    let mut files = SimpleFiles::new();
    files.add("src", example);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();

    let mut lexer = lexer::Lexer::new(example);
    if let Err(err) = lexer.lex() {
        let diagnostic = err.to_diagnostic(&files);
        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        return;
    }

    // let mut lexer_copy = lexer.clone();

    let mut parser = parser::Parser::new(lexer);
    let result = parser.parse();
    if let Err(err) = result {
        // println!("--- Tokens:");
        // loop {
        //     let token = lexer_copy.next();
        //     print!("{:#?}\n", token);

        //     if let lexer::Token::EOF { .. } = token {
        //         break;
        //     }
        // }

        parser.display_error(err);
        return;
    }

    let ast = result.unwrap();

    // println!("\n--- AST:");
    // println!("{:#?}", result.unwrap());

    let mut tail = Interpreter::new(ast);
    let result = tail.run();

    println!("{}", result);
}
