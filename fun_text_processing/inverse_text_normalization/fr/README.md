# Note on French spelling

Due to a 1990 orthographic reform, there are currently two conventions for written French numbers:

1. **Reformed** All composite words are joined by a hyphen: 
e.g. `1122 -> mille-cent-vingt-deux`

2. **Traditional** Hyphenation only occurs (with exception) for numbers from 17 to 99 (inclusive):
e.g. `1122 -> mille cent vingt-deux`

As available training data for upstream ASR will vary in use of convention, NeMo's French ITN accomodates either style for normalization e.g.

```
	python inverse_normalize.py "mille-cent-vingt-deux" --language="fr"  --> 1122
	python inverse_normalize.py "mille cent vingt-deux" --language="fr"  --> 1122
```

As a result, there exists some ambiguity in the case of currency conversions, namely minor denominations of the dollar e.g.

```
	300 -> "trois-cents" # Reformed spelling
	300 -> "trois cents" # Traditional spelling
	3 ¢ -> "trois cents" # Valid for both
```

Cardinals take priority in such cases. 

```
python inverse_normalize.py "trois cents" --language="fr" -> 300
``` 
