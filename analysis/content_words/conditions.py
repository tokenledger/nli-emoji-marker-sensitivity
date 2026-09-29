"""Condition registry for the content-word two-word controls (hypothesis suffix)."""
NEW_CONDITIONS = [
    # condition_id, phrase, group
    ("c24_formal_content_in_truth", "in truth", "formal_content_noun"),
    ("c25_formal_content_in_essence", "in essence", "formal_content_noun"),
    ("c26_formal_content_in_short", "in short", "formal_content_noun"),
    ("c27_formal_content_in_reality", "in reality", "formal_content_noun"),
    ("c28_religious_heaven_knows", "heaven knows", "religious_noun_commitment"),
    ("c29_religious_lord_knows", "lord knows", "religious_noun_commitment"),
    ("c30_religious_by_heaven", "by heaven", "religious_noun_commitment"),
    ("c31_religious_good_heavens", "good heavens", "religious_noun_commitment"),
    ("c32_informal_content_for_real", "for real", "informal_content_marker"),
]
CONTENT_WORD = {"in truth": "truth", "in essence": "essence", "in short": "short", "in reality": "reality",
                "heaven knows": "heaven", "lord knows": "lord", "by heaven": "heaven", "good heavens": "heavens",
                "for real": "real"}
# Existing crossed conditions shown alongside (from the archived predictions).
REFERENCE_CONDITIONS = [
    ("c05_informal_on_god", "on god", "informal_full_phrase"),
    ("c13_component_god", "god", "component_ablation"),
    ("c18_formal_in_fact", "in fact", "formal_control"),
    ("c19_random_nearby", "nearby", "random_control"),
    ("c17_formal_seriously", "seriously", "formal_control"),
]
