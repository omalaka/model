# From omalaka.net

Sentence pairs written, translated and checked by speakers on
[omalaka.net](https://omalaka.net), exported here in the same format as
[../README.md](../README.md) describes. Don't edit these files by hand: they
are replaced by each export. Corrections go through the website.

| File | What | Licence |
|---|---|---|
| `kj.tsv`, `ng.tsv` | Sentences written on omalaka.net in Oshikwanyama or Oshindonga, translated into English by a second contributor, checked by at least one more (two for `split=test`) | CC BY 4.0, credited to `translator` and `reviewed_by`; `note` names the writer if that's someone else |
| `kj-from-english.tsv`, `ng-from-english.tsv` | Short everyday English sentences from [Tatoeba](https://tatoeba.org), translated into Oshikwanyama or Oshindonga by an omalaka.net contributor and checked by another | Oshiwambo side: CC BY 4.0, by `translator`. English side: the Tatoeba sentence, [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/), whose number and author are in `note`. Credit both when you reuse a row |

Ids are stable and never reused (`kj-000001` for the first file, `kj-en-000001`
for the second). Translations from English are always `split=train`: they
read like translations, so they never go into dev or test sets.

Contributors agreed on omalaka.net to write in their own words, without
copying or machine translation, and to the CC BY 4.0 licence. People under 18
are credited by a nickname; anyone may choose "Anonymous".

In the training pipeline the first two files are part of the `community`
dataset; the `-from-english` files are the separate dataset
`community_from_english` (kind `translated`: training only).
