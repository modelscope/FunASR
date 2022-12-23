# Localization Note

Depending on locale, Spanish number strings will vary in formatting. In the EU and South American countries, it is common to use a period (".") or space to delineate groupings of three
digits. e.g.
	`1.000.000` -> "un millón"
	`1 000 000` -> "un millón"

and commas (",") to seperate cardinal and decimal strings. e.g.

	`1,00` -> "uno coma cero cero"

While Central and Northern America will use commas (",") to delineate groupings of three digits, e.g.
	`1,000,000` -> "un millón"

and periods (".") to seperate cardinal and decimal strings. e.g.

	`1.00` -> "uno coma cero cero"

As inclusion of both forms will create inherrent ambiguity for verbalization, this module defaults to the former formatting (periods for cardinal delineation and commas for decimals).

To toggle the alternate formatting, you may edit the `LOCALIZATION` variable in `fun_text_processing.text_normalization.es.__init__` with the value of `'am'`. This will perform necessary
adjustments to all affected classes.