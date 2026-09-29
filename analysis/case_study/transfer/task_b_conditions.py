"""Conditions for the independent-data confirmation (hypothesis suffix, crossed-evaluation rule)."""
CONDITIONS = [
    ("c00_clean", "", "clean"),
    ("c05_informal_on_god", "on god", "informal_full_phrase"),
    ("c13_component_god", "god", "component_ablation"),
    ("c12_component_on", "on", "component_ablation"),
    ("c09_informal_no_cap", "no cap", "informal_full_phrase"),
    ("c08_informal_deadass", "deadass", "informal_single_token"),
    ("c04_informal_real_talk", "real talk", "informal_full_phrase"),
    ("c18_formal_in_fact", "in fact", "formal_control"),
    ("c19_random_nearby", "nearby", "random_control"),
    ("c17_formal_seriously", "seriously", "formal_control"),
    ("c16_formal_honestly", "honestly", "formal_control"),
    ("c21_random_in_the", "in the", "random_control"),
]
CONTROL_IDS = ["c18_formal_in_fact", "c19_random_nearby", "c17_formal_seriously", "c16_formal_honestly", "c21_random_in_the"]
ON_GOD = "c05_informal_on_god"
DATASETS = {"sick": "sick_test.jsonl", "anli": "anli_dev.jsonl"}
