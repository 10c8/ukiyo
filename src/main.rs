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

use crate::lexer::ToDiagnostic;

fn main() {
    //     let example = r#"
    // fruits -> ["apples", "bananas", "oranges", "limes"]
    // flavors -> {
    //   "fruit"     = ["sweet", "sour", "bitter"],
    //   "vegetable" = ["bitter", "salty", "umami"],
    //   "meat"      = ["salty", "umami", "sweet"],
    // }

    // get_color fruit ->
    //   color -> case fruit of
    //     "apples"            => "red"
    //     "bananas" | "limes" => "yellow"
    //     "oranges"           => "orange"
    //     _                   => gen %% { "ctx" = ["give me the fruit color"], "temp" = 0.3 }

    //   $"{fruit} are {color}"

    // fruit_colors ->
    //   case (len fruits) of
    //     1 => get_color (head fruits)
    //     _ =>
    //       start -> join ((init fruits) |f| get_color f) ", "
    //       end   -> get_color (last fruits)

    //       $"{start} and {end}"
    // "#;

    let example = r#"
const config -> {
    "ctx" = @"
    This agent generates a random RPG character.
    The character has a name, a class, and an item inventory.
    Its class is one of the following: warrior, mage, rogue, or cleric.
    It has a random number of items in its inventory.
    Items are one of the following: sword, axe, staff, dagger, or wand.
    "@,
    "temp" = 0.7,
}

const class -> gen "The character is a " config

const name -> gen $"The {class} is called " config

gen_item ->
    const item_name -> gen $"Give me an item name:" config
    item_desc -> gen $"Describe the item called \"{item_name}\":" config

    $@"
    {
      "name": "{item_name}",
      "description": "{item_desc}"
    }
    "@

inventory -> join (1..=(rand 1 5) |_| gen_item) ",\n"
"#;

    let mut files = SimpleFiles::new();
    files.add("src", example);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();

    let mut lexer = lexer::Lexer::new(example);
    let result = lexer.lex();
    if let Err(err) = result {
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

    println!("\n--- AST:");
    println!("{:#?}", result.unwrap());
}
