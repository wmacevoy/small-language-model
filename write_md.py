from vocabulary import vocabulary

md = """# Ing Say: A minimalist pidgin English.

## Introduction

The purpose of Ing Say is to communicate in a simplified world, where the universe is a small connected graph of locations occupied by objects and entities.

The alphabet is lowercase, and punctuation is period(.), question (?), and exclamation (!).  Words can be grouped with quotes (any kind, including parenthesis) for composite concepts.

## Lesson 1: Basics

Sentence structure is always simple:

Subject + Verb + (Object)

Examples:

* i eat food. (I eat food - just a fact about me)
* they like cat. (they like cat - also a fact)

## Lesson 2: Time Words

Use explicit verb prefix words to show time:

* ed (past)
* ing (present continuous)
* will (future)

Examples:

* i ed eat food. (I ate food)
* they will like cat. (They will like the cat)
* we ing friend. (We are currenly friends)

## Lesson 3: 

no represents the set compliment, negation or false.

* i no eat meat. (i do not eat meat)
* i eat no meat. (i eat anything-presumbaly food-that is not meat)

## Lesson 4: Yes or No Questions

Yes or no questions are a statement followed by ha?

* ha you ing move home? (Are you moving home now?)

A reponse to a yes or no question can be:

* yes.
* no.
* ha.  (I do not know or am uncertain)
* no ha. (Invalid question)

## Lesson 5: Placeholder questions

Ha can be a placeholder in a statement to turn it into a question.

* you ing move ha?  (Where are you moving to?)

A response can simply be the placeholder

* home.
* no ha.  (invalid question, maybe I am not moving)

## Lesson 6: Go is desire or imperitive to change state.

* i go left.
* they go eat.

A state change may require resources, so use those

* i use bike go home.
* i use key go open door.

## Lesson 7: Vocabulary

"""

words_per_line = 15
for category, words in vocabulary.items():
    md += f"* {category.capitalize()}: "
    for i, word in enumerate(words, 1):
        md += word
        if i != len(words):
            md += ", "
        if i % words_per_line == 0:
            md += "\n" + " " * (len(category) + 2)
    md += '\n'

md += """
Example sentences:

they ing move right. (They are moving to the right.)

all bird ed fly south. (the birds flew south)

you will move at i. ha? (Will you come towards me?)

you will eat ha? (What will you eat?)"""

with open("pidgin.md", "w") as file:
    file.write(md)