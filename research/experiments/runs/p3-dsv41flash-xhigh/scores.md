# Scores — p3-dsv41flash-xhigh (2026-10-19)

Cases scored: 169 (ok 169) | harness_sha256: `d8521309abb11d3648d246094690ec6c513b6bc02b252fff68bfb4d90a6512cc` | model: opencode-go/deepseek-v4.1-flash

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F02 | F | 0.6097 | 11.225 | 0.5257 | v1 | 0.6097 | ✓ | yes | 0 | budget | — | — | — |
| F06 | F | 0.7958 | 7.4833 | 0.6774 | v2 | 0.7958 | ✓ | yes | 0 | none | — | — | — |
| F07 | F | 0.4595 | 285.7224 | 0.2828 | v2 | 0.4595 | ✓ | no | 5 | localization | — | — | — |
| F09 | F | 0.8562 | 104.2708 | 0.5527 | v2 | 0.8562 | ✓ | no | 25.729 | none | — | — | — |
| F11 | F | 0.7975 | 4 | 0.8541 | v1 | 0.7975 | ✓ | yes | 0 | budget | — | — | — |
| F12 | F | 0.0001 | 97.009 | 0.0014 | v1 | 0.0001 | ✗ | no | 22.672 | localization | wrong_structure | — | — |
| F13 | F | 0.7641 | 17 | 0.6158 | v3 | 0.801 | ✗ | yes | 0 | budget | — | — | — |
| F16 | F | 0.0005 | 31.209 | 0.011 | v2 | 0.0005 | ✓ | no | 19 | localization | wrong_structure | — | — |
| F17 | F | 0.9198 | 3.6056 | 0.7925 | v3 | 0.9198 | ✓ | yes | 0 | none | — | — | — |
| F18 | F | 0 | 123.8144 | 0 | v2 | 0.2532 | ✗ | no | 21.401 | localization | wrong_structure | — | — |
| F19 | F | 0.8766 | 4.1231 | 0.8045 | v1 | 0.8766 | ✓ | yes | 0 | none | — | — | — |
| F20 | F | 0.0001 | 51.8015 | 0.0232 | v2 | 0.0001 | ✓ | no | 18.974 | localization | wrong_structure | — | — |
| F21 | F | 0.7048 | 45.8476 | 0.2188 | v2 | 0.7048 | ✓ | yes | 0 | none | — | — | — |
| F22 | F | 0 | — | — | — | 0 | ✓ | no | — | localization | empty_pred | — | — |
| F24 | F | 0.8708 | 8.775 | 0.8011 | v2 | 0.8708 | ✓ | yes | 0 | none | — | — | — |
| F26 | F | 0.4197 | 75.5315 | 0.1949 | v2 | 0.4197 | ✓ | yes | 0 | delineation | — | — | — |
| F28 | F | 0 | 116.9872 | 0 | v3 | 0.0193 | ✗ | no | 62.586 | localization | wrong_structure | — | — |
| F29 | F | 0.7734 | 10.6301 | 0.5859 | v2 | 0.8399 | ✗ | yes | 0 | none | — | — | — |
| F31 | F | 0.8511 | 13.4536 | 0.379 | v1 | 0.8511 | ✓ | yes | 0 | none | — | — | — |
| F33 | F | 0 | — | — | v1 | 0 | ✓ | no | 18.221 | localization | empty_pred | — | — |
| F35 | F | 0 | 164.0236 | 0 | v1 | 0 | ✓ | no | 147.34 | localization | wrong_structure | — | — |
| F36 | F | 0.6161 | 32.1302 | 0.4493 | v2 | 0.6161 | ✓ | no | 5 | budget | — | — | — |
| F37 | F | 0.7713 | 9.4868 | 0.2485 | v2 | 0.7713 | ✓ | yes | 0 | none | — | — | — |
| F38 | F | 0.6747 | 28.7228 | 0.4872 | v3 | 0.6747 | ✓ | no | 16.401 | none | — | — | — |
| F39 | F | 0.7801 | 29.0129 | 0.3781 | v2 | 0.7801 | ✓ | yes | 0 | none | — | — | — |
| F41 | F | 0.9237 | 3.3166 | 0.84 | v2 | 0.9237 | ✓ | yes | 0 | budget | — | — | — |
| F42 | F | 0 | 95.7943 | 0 | v1 | 0 | ✓ | no | 64.645 | localization | wrong_structure | — | — |
| F43 | F | 0.7381 | 22.1133 | 0.4717 | v1 | 0.7381 | ✓ | yes | 0 | budget | — | — | — |
| F45 | F | 0.8116 | 29.6985 | 0.4723 | v4 | 0.8116 | ✓ | yes | 0 | none | — | — | — |
| F46 | F | 0.4773 | 41.5933 | 0.2554 | v3 | 0.4773 | ✓ | no | 4 | localization | — | — | — |
| F48 | F | 0 | 87.7838 | 0.0024 | v1 | 0 | ✓ | no | 65.399 | localization | wrong_structure | — | — |
| F51 | F | 0 | 64.5446 | 0 | v1 | 0.0656 | ✗ | yes | 0 | delineation | wrong_structure | — | — |
| F53 | F | 0.0002 | 75.4255 | 0.0027 | v2 | 0.0002 | ✗ | no | 13.928 | localization | wrong_structure | — | — |
| F55 | F | 0 | 54.5711 | 0 | v1 | 0 | ✓ | no | 43.012 | localization | wrong_structure | — | — |
| L01 | L | 0.9001 | 18.0748 | 0.6693 | v1 | 0.9001 | ✓ | yes | 0 | budget | — | — | — |
| L02 | L | 0 | 134.3652 | 0 | v1 | 0 | ✓ | no | 19.95 | localization | wrong_structure | — | — |
| L03 | L | 0.9664 | 3.1623 | 0.8605 | v1 | 0.9664 | ✓ | yes | 0 | harness | — | — | — |
| L04 | L | 0.9089 | 6.0828 | 0.7264 | v3 | 0.9089 | ✓ | yes | 0 | none | — | — | — |
| L05 | L | 0.9279 | 10.0995 | 0.7309 | v1 | 0.9289 | ✗ | yes | 0 | budget | — | — | — |
| L07 | L | 0.9032 | 10.0995 | 0.742 | v5 | 0.9032 | ✓ | yes | 0 | budget | — | — | — |
| L08 | L | 0 | 142.1077 | 0 | v2 | 0.0001 | ✗ | no | 24.515 | localization | wrong_structure | — | — |
| L09 | L | 0.154 | 73.4248 | 0.2905 | v3 | 0.154 | ✓ | no | 4 | localization | — | — | — |
| L10 | L | 0.9278 | 7.8102 | 0.6659 | v1 | 0.9278 | ✓ | yes | 0 | budget | — | — | — |
| L11 | L | 0.8722 | 19.2094 | 0.7039 | v2 | 0.9515 | ✗ | yes | 0 | none | — | — | — |
| L12 | L | 0.8755 | 11.7898 | 0.6106 | v3 | 0.8755 | ✓ | yes | 0 | none | — | — | — |
| L13 | L | 0 | 79.6552 | 0 | v2 | 0.0178 | ✗ | no | 58.489 | localization | wrong_structure | — | — |
| L16 | L | 0.9399 | 4 | 0.882 | v2 | 0.9399 | ✓ | yes | 0 | none | — | — | — |
| L17 | L | 0.8934 | 11.1803 | 0.6483 | v1 | 0.9161 | ✗ | yes | 0 | budget | — | — | — |
| L18 | L | 0.8887 | 8.5498 | 0.7022 | v4 | 0.8887 | ✓ | no | 31.241 | none | — | — | — |
| L21 | L | 0.8809 | 22.1133 | 0.685 | v1 | 0.8916 | ✗ | yes | 0 | none | — | — | — |
| L22 | L | 0.3748 | 49.2646 | 0.1819 | v1 | 0.8276 | ✗ | yes | 0 | delineation | — | — | — |
| L23 | L | 0.7143 | 30.1496 | 0.4638 | v2 | 0.7143 | ✓ | no | 29.833 | none | — | — | — |
| L24 | L | 0.7151 | 22.7596 | 0.257 | v1 | 0.7151 | ✓ | yes | 0 | budget | — | — | — |
| L25 | L | 0 | 95.6922 | 0 | v1 | 0 | ✓ | no | 52.64 | localization | wrong_structure | — | — |
| L26 | L | 0.6335 | 31.9531 | 0.1327 | v2 | 0.6335 | ✓ | yes | 0 | budget | — | — | — |
| L27 | L | 0.8793 | 13.4536 | 0.6169 | v1 | 0.9035 | ✗ | yes | 0 | budget | — | — | — |
| L28 | L | 0.8828 | 13.4907 | 0.6669 | v1 | 0.8854 | ✗ | yes | 0 | budget | — | — | — |
| L29 | L | 0.6585 | 24.9018 | 0.5014 | v1 | 0.676 | ✗ | yes | 0 | budget | — | — | — |
| L31 | L | 0.926 | 8.2462 | 0.7482 | v2 | 0.926 | ✓ | yes | 0 | budget | — | — | — |
| L32 | L | 0.8119 | 17.6068 | 0.4662 | v2 | 0.8119 | ✓ | yes | 0 | none | — | — | — |
| L34 | L | 0.7488 | 28.6182 | 0.2488 | v1 | 0.7488 | ✓ | yes | 0 | none | — | — | — |
| L36 | L | 0.5431 | 67.6535 | 0.3228 | v1 | 0.5431 | ✓ | yes | 0 | none | — | — | — |
| L37 | L | 0.9621 | 2.8284 | 0.9026 | v1 | 0.9621 | ✓ | yes | 0 | budget | — | — | — |
| L39 | L | 0.9229 | 13.6015 | 0.6139 | v3 | 0.9229 | ✓ | yes | 0 | none | — | — | — |
| L40 | L | 0 | 223.5502 | 0 | v1 | 0 | ✓ | no | 41.024 | localization | wrong_structure | — | — |
| L41 | L | 0.9395 | 8.0623 | 0.7889 | v1 | 0.9395 | ✓ | yes | 0 | none | — | — | — |
| L42 | L | 0.958 | 3.7417 | 0.855 | v2 | 0.958 | ✓ | yes | 0 | none | — | — | — |
| L44 | L | 0.5592 | 36.7287 | 0.0877 | v1 | 0.5592 | ✓ | yes | 0 | budget | — | — | — |
| M02 | M | 0.9465 | 3 | 0.894 | v1 | 0.9465 | ✓ | yes | 0 | none | — | — | — |
| M04 | M | 0.8509 | 16.1864 | 0.7452 | v3 | 0.8957 | ✗ | yes | 0 | none | — | — | — |
| M05 | M | 0 | 69.2387 | 0.0004 | v1 | 0.866 | ✗ | yes | 0 | delineation | — | — | — |
| M06 | M | 0.6843 | 53.8609 | 0.4972 | v1 | 0.6843 | ✓ | yes | 0 | none | — | — | — |
| M10 | M | 0.9591 | 2 | 0.9789 | v1 | 0.9591 | ✓ | yes | 0 | none | — | — | — |
| M11 | M | 0.9354 | 3.4641 | 0.8765 | v1 | 0.9354 | ✓ | yes | 0 | none | — | — | — |
| M12 | M | 0.8952 | 3.1623 | 0.8532 | v2 | 0.8952 | ✓ | yes | 0 | none | — | — | — |
| M13 | M | 0.874 | 9 | 0.7502 | v3 | 0.874 | ✓ | no | 1 | none | — | — | — |
| M14 | M | 0.8101 | 6.7082 | 0.7132 | v1 | 0.8834 | ✗ | yes | 0 | none | — | — | — |
| M15 | M | 0.8796 | 9.1104 | 0.5835 | v1 | 0.8796 | ✓ | yes | 0 | none | — | — | — |
| M16 | M | 0.9172 | 3.6668 | 0.8458 | v1 | 0.9172 | ✓ | yes | 0 | none | — | — | — |
| M17 | M | 0.6459 | 31.4643 | 0.5033 | v2 | 0.6528 | ✗ | yes | 0 | budget | — | — | — |
| M18 | M | 0.9438 | 2 | 0.9502 | v1 | 0.9438 | ✓ | yes | 0 | none | — | — | — |
| M19 | M | 0.9195 | 4 | 0.7787 | v1 | 0.9482 | ✗ | yes | 0 | none | — | — | — |
| M22 | M | 0.723 | 33.3617 | 0.4009 | v1 | 0.723 | ✓ | yes | 0 | none | — | — | — |
| M23 | M | 0.7667 | 29.2916 | 0.1769 | v2 | 0.7667 | ✓ | yes | 0 | budget | — | — | — |
| M24 | M | 0.7318 | 41.7612 | 0.4113 | v1 | 0.8215 | ✗ | yes | 0 | budget | — | — | — |
| M25 | M | 0.3367 | 74.6391 | 0.105 | v2 | 0.3367 | ✓ | yes | 0 | delineation | — | — | — |
| M26 | M | 0.9584 | 1.7321 | 0.9708 | v1 | 0.9584 | ✓ | yes | 0 | none | — | — | — |
| M28 | M | 0.935 | 2.8284 | 0.9211 | v2 | 0.935 | ✓ | yes | 0 | none | — | — | — |
| M30 | M | 0.9412 | 2 | 0.9718 | v2 | 0.9597 | ✗ | yes | 0 | none | — | — | — |
| M31 | M | 0.7973 | 44.1135 | 0.5886 | v3 | 0.8968 | ✗ | yes | 0 | budget | — | — | — |
| M32 | M | 0.9247 | 5 | 0.6891 | v1 | 0.9285 | ✗ | yes | 0 | budget | — | — | — |
| M34 | M | 0 | 136.3429 | 0 | v1 | 0 | ✓ | no | 110.336 | localization | wrong_structure | — | — |
| M35 | M | 0.9005 | 5.3852 | 0.6349 | v2 | 0.9005 | ✓ | yes | 0 | none | — | — | — |
| M36 | M | 0.9343 | 2.2361 | 0.927 | v1 | 0.9343 | ✓ | yes | 0 | budget | — | — | — |
| M37 | M | 0.7597 | 34.0147 | 0.5567 | v2 | 0.8591 | ✗ | yes | 0 | none | — | — | — |
| M38 | M | 0.6647 | 17.72 | 0.3605 | v1 | 0.6647 | ✓ | yes | 0 | budget | — | — | — |
| M39 | M | 0 | 130.3956 | 0 | v1 | 0 | ✓ | no | 102.206 | localization | wrong_structure | — | — |
| M40 | M | 0.91 | 6 | 0.7975 | v2 | 0.91 | ✓ | yes | 0 | budget | — | — | — |
| M41 | M | 0 | 128.74 | 0 | v1 | 0 | ✓ | no | 54.562 | localization | wrong_structure | — | — |
| M42 | M | 0 | 69.6922 | 0 | v1 | 0 | ✓ | no | 52.259 | localization | wrong_structure | — | — |
| M43 | M | 0.6001 | 10.198 | 0.5461 | v2 | 0.6001 | ✓ | yes | 0 | budget | — | — | — |
| M44 | M | 0.9379 | 4.4721 | 0.7289 | v2 | 0.9418 | ✗ | yes | 0 | none | — | — | — |
| N03 | N | 0.8231 | 7.2246 | 0.869 | v2 | 0.8728 | ✗ | yes | 0 | none | — | — | — |
| N04 | N | 0.9432 | 1 | 0.9898 | v1 | 0.9459 | ✗ | yes | 0 | none | — | — | — |
| N05 | N | 0.8074 | 8.3066 | 0.7893 | v1 | 0.8443 | ✗ | yes | 0 | none | — | — | — |
| N06 | N | 0 | 74.1809 | 0 | v1 | 0 | ✓ | no | 66.611 | localization | wrong_structure | — | — |
| N07 | N | 0.9523 | 2 | 0.9643 | v1 | 0.9523 | ✓ | yes | 0 | none | — | — | — |
| N08 | N | 0.8934 | 2.4495 | 0.9149 | v1 | 0.8934 | ✓ | yes | 0 | budget | — | — | — |
| N09 | N | 0.8107 | 11.3578 | 0.4143 | v1 | 0.8129 | ✗ | yes | 0 | none | — | — | — |
| N11 | N | 0.8524 | 1.4142 | 1 | v2 | 0.8524 | ✓ | yes | 0 | budget | — | — | — |
| N13 | N | 0.9442 | 2.2361 | 0.9486 | v2 | 0.9442 | ✓ | yes | 0 | none | — | — | — |
| N14 | N | 0 | 58.9067 | 0 | v1 | 0 | ✓ | no | 29.883 | localization | wrong_structure | — | — |
| N15 | N | 0.7525 | 21.5639 | 0.5344 | v1 | 0.7525 | ✓ | yes | 0 | none | — | — | — |
| N16 | N | 0.9159 | 4.1231 | 0.7925 | v1 | 0.9354 | ✗ | yes | 0 | none | — | — | — |
| N17 | N | 0.893 | 2.2361 | 0.937 | v1 | 0.893 | ✓ | yes | 0 | none | — | — | — |
| N19 | N | 0.9124 | 3 | 0.9128 | v2 | 0.9124 | ✓ | yes | 0 | none | — | — | — |
| N20 | N | 0.8097 | 5.3852 | 0.7531 | v2 | 0.8097 | ✓ | yes | 0 | none | — | — | — |
| N22 | N | 0.833 | 21.4091 | 0.5815 | v3 | 0.833 | ✗ | yes | 0 | none | — | — | — |
| N23 | N | 0.0567 | 152.1085 | 0.0144 | v1 | 0.0583 | ✗ | yes | 0 | delineation | — | — | — |
| N24 | N | 0.9528 | 5.7446 | 0.6956 | v2 | 0.9542 | ✗ | yes | 0 | none | — | — | — |
| N27 | N | 0.701 | 29.1376 | 0.4166 | v1 | 0.7064 | ✗ | yes | 0 | none | — | — | — |
| N28 | N | 0.917 | 191.7441 | 0.7298 | v5 | 0.917 | ✓ | yes | 0 | budget | — | — | — |
| N32 | N | 0.8904 | 5.3852 | 0.7559 | v1 | 0.923 | ✗ | yes | 0 | budget | — | — | — |
| N33 | N | 0.9509 | 2 | 0.9548 | v1 | 0.9509 | ✓ | yes | 0 | none | — | — | — |
| N35 | N | 0.8312 | 25.8844 | 0.3981 | v2 | 0.8312 | ✓ | yes | 0 | budget | — | — | — |
| N36 | N | 0.8838 | 8.1854 | 0.8073 | v2 | 0.8838 | ✓ | yes | 0 | budget | — | — | — |
| N37 | N | 0.7641 | 25.017 | 0.7064 | v1 | 0.7641 | ✓ | yes | 0 | none | — | — | — |
| N38 | N | 0.2864 | 32.45 | 0.2851 | v1 | 0.2864 | ✓ | yes | 0 | delineation | — | — | — |
| N39 | N | 0.8754 | 2 | 0.958 | v1 | 0.8754 | ✓ | yes | 0 | none | — | — | — |
| N41 | N | 0 | 45.8148 | 0 | v1 | 0 | ✓ | no | 40.324 | localization | wrong_structure | — | — |
| N42 | N | 0.9216 | 2.2361 | 0.9458 | v2 | 0.9216 | ✓ | yes | 0 | none | — | — | — |
| N44 | N | 0 | 48.3823 | 0.0118 | v1 | 0 | ✓ | no | 31.064 | localization | wrong_structure | — | — |
| N45 | N | 0.9105 | 3 | 0.9027 | v2 | 0.9105 | ✓ | yes | 0 | none | — | — | — |
| N46 | N | 0.9487 | 3.1623 | 0.8768 | v2 | 0.9504 | ✗ | yes | 0 | none | — | — | — |
| N47 | N | 0.9274 | 1 | 0.995 | v1 | 0.9274 | ✓ | yes | 0 | none | — | — | — |
| N48 | N | 0.911 | 9.2195 | 0.8429 | v1 | 0.9196 | ✗ | yes | 0 | none | — | — | — |
| V01 | V | 0 | 43.3405 | 0 | v1 | 0 | ✓ | no | 26.476 | localization | wrong_structure | — | — |
| V02 | V | 0.8021 | 3 | 0.9087 | v3 | 0.8115 | ✗ | yes | 0 | none | — | — | — |
| V06 | V | 0.8712 | 2.4281 | 0.9148 | v1 | 0.9216 | ✗ | yes | 0 | none | — | — | — |
| V07 | V | 0.008 | 18.4662 | 0.14 | v2 | 0.008 | ✓ | no | 7.81 | localization | wrong_structure | — | — |
| V08 | V | 0 | 97.3717 | 0 | v1 | 0 | ✓ | no | 52.745 | localization | wrong_structure | — | — |
| V09 | V | 0.8956 | 2.2361 | 0.9413 | v1 | 0.8983 | ✗ | yes | 0 | none | — | — | — |
| V14 | V | 0.8859 | 5.3852 | 0.7006 | v3 | 0.8859 | ✓ | yes | 0 | budget | — | — | — |
| V15 | V | 0 | 61.6685 | 0.0037 | v1 | 0 | ✓ | no | 51.049 | localization | wrong_structure | — | — |
| V16 | V | 0.7387 | 11.3578 | 0.7105 | v1 | 0.7613 | ✗ | yes | 0 | none | — | — | — |
| V18 | V | 0.0153 | 15.582 | 0.1243 | v1 | 0.0171 | ✗ | no | 6.164 | localization | wrong_structure | — | — |
| V24 | V | 0.0043 | 62.2575 | 0.0946 | v3 | 0.0043 | ✓ | no | 15.524 | localization | wrong_structure | — | — |
| V25 | V | 0 | 292.0428 | 0 | v1 | 0 | ✓ | no | 168.834 | localization | wrong_structure | — | — |
| V33 | V | 0.8995 | 7.4833 | 0.6512 | v7 | 0.8995 | ✓ | no | 5.385 | none | — | — | — |
| V34 | V | 0.8377 | 13.6015 | 0.5721 | v1 | 0.8447 | ✗ | yes | 0 | none | — | — | — |
| V35 | V | 0.9351 | 3.3166 | 0.8904 | v1 | 0.9351 | ✓ | yes | 0 | budget | — | — | — |
| V36 | V | 0.7433 | 30.2985 | 0.3079 | v2 | 0.7913 | ✗ | yes | 0 | budget | — | — | — |
| V38 | V | 0.733 | 29.1204 | 0.4469 | v1 | 0.733 | ✓ | yes | 0 | budget | — | — | — |
| V39 | V | 0.9492 | 3 | 0.8791 | v2 | 0.9492 | ✓ | yes | 0 | none | — | — | — |
| V41 | V | 0.8829 | 1.4142 | 0.9878 | v4 | 0.8829 | ✓ | yes | 0 | none | — | — | — |
| V42 | V | 0.8691 | 2 | 0.9596 | v2 | 0.8691 | ✓ | yes | 0 | none | — | — | — |
| V48 | V | 0.8935 | 3 | 0.9168 | v3 | 0.8935 | ✓ | yes | 0 | none | — | — | — |
| V49 | V | 0 | 47.9573 | 0 | v1 | 0 | ✓ | no | 19.131 | localization | wrong_structure | — | — |
| V50 | V | 0.8036 | 2.8284 | 0.9094 | v1 | 0.8044 | ✗ | yes | 0 | budget | — | — | — |
| V51 | V | 0.8401 | 2.4495 | 0.8913 | v1 | 0.8417 | ✗ | yes | 0 | none | — | — | — |
| V52 | V | 0 | 66.9955 | 0.0004 | v1 | 0 | ✗ | no | 30.676 | localization | wrong_structure | — | — |
| V53 | V | 0 | 218.9897 | 0 | v1 | 0 | ✓ | no | 183.469 | localization | wrong_structure | — | — |
| V56 | V | 0.838 | 3 | 0.9135 | v4 | 0.8382 | ✗ | yes | 0 | budget | — | — | — |
| V58 | V | 0.8711 | 3 | 0.9165 | v3 | 0.8711 | ✓ | yes | 0 | budget | — | — | — |
| V59 | V | 0.9513 | 1 | 1 | v2 | 0.9513 | ✓ | yes | 0 | none | — | — | — |
| V60 | V | 0.8652 | 1.7321 | 0.9651 | v1 | 0.8664 | ✗ | yes | 0 | none | — | — | — |
| V61 | V | 0.8371 | 4.1231 | 0.8798 | v3 | 0.8371 | ✓ | yes | 0 | none | — | — | — |
| V63 | V | 0.6383 | 15.5242 | 0.4462 | v1 | 0.6383 | ✓ | yes | 0 | none | — | — | — |
| V64 | V | 0 | — | — | — | 0 | ✓ | no | — | localization | empty_pred | — | — |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 34 | 0.456 | 0.613 | 0.507 | -0.051 | — | — |
| L | 34 | 0.684 | 0.877 | 0.744 | -0.060 | — | — |
| M | 34 | 0.708 | 0.862 | 0.823 | -0.115 | — | — |
| N | 34 | 0.732 | 0.880 | 0.748 | -0.016 | — | — |
| V | 33 | 0.564 | 0.804 | 0.584 | -0.020 | — | — |
| **overall** | 169 | 0.629 | 0.810 | 0.681 | -0.052 | — | — |

## Notes

- F12 (F): flags wrong_structure (dice=0.0001, tag=localization).
- F16 (F): flags wrong_structure (dice=0.0005, tag=localization).
- F18 (F): flags wrong_structure (dice=0, tag=localization).
- F20 (F): flags wrong_structure (dice=0.0001, tag=localization).
- F22 (F): flags empty_pred (dice=0, tag=localization).
- F28 (F): flags wrong_structure (dice=0, tag=localization).
- F33 (F): flags empty_pred (dice=0, tag=localization).
- F35 (F): flags wrong_structure (dice=0, tag=localization).
- F42 (F): flags wrong_structure (dice=0, tag=localization).
- F48 (F): flags wrong_structure (dice=0, tag=localization).
- F51 (F): flags wrong_structure (dice=0, tag=delineation).
- F53 (F): flags wrong_structure (dice=0.0002, tag=localization).
- F55 (F): flags wrong_structure (dice=0, tag=localization).
- L02 (L): flags wrong_structure (dice=0, tag=localization).
- L08 (L): flags wrong_structure (dice=0, tag=localization).
- L13 (L): flags wrong_structure (dice=0, tag=localization).
- L25 (L): flags wrong_structure (dice=0, tag=localization).
- L40 (L): flags wrong_structure (dice=0, tag=localization).
- M34 (M): flags wrong_structure (dice=0, tag=localization).
- M39 (M): flags wrong_structure (dice=0, tag=localization).
- M41 (M): flags wrong_structure (dice=0, tag=localization).
- M42 (M): flags wrong_structure (dice=0, tag=localization).
- N06 (N): flags wrong_structure (dice=0, tag=localization).
- N14 (N): flags wrong_structure (dice=0, tag=localization).
- N41 (N): flags wrong_structure (dice=0, tag=localization).
- N44 (N): flags wrong_structure (dice=0, tag=localization).
- V01 (V): flags wrong_structure (dice=0, tag=localization).
- V07 (V): flags wrong_structure (dice=0.008, tag=localization).
- V08 (V): flags wrong_structure (dice=0, tag=localization).
- V15 (V): flags wrong_structure (dice=0, tag=localization).
- V18 (V): flags wrong_structure (dice=0.0153, tag=localization).
- V24 (V): flags wrong_structure (dice=0.0043, tag=localization).
- V25 (V): flags wrong_structure (dice=0, tag=localization).
- V49 (V): flags wrong_structure (dice=0, tag=localization).
- V52 (V): flags wrong_structure (dice=0, tag=localization).
- V53 (V): flags wrong_structure (dice=0, tag=localization).
- V64 (V): flags empty_pred (dice=0, tag=localization).

