#!/bin/bash

# synth zh spker: SSB0009 SSB1891 SSB0005 / SSB0603 SSB0710 SSB0966
python3 synthesize.py --text "醫生說是劇烈運動過量所至" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0009 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "醫生說是劇烈運動過量所至" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB1891 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "醫生說是劇烈運動過量所至" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0005 --text_lang "zh" --lang "zh"

python3 synthesize.py --text "你喜歡我的帽子嗎" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0603 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "你喜歡我的帽子嗎" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0710 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "你喜歡我的帽子嗎" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0966 --text_lang "zh" --lang "zh"

python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0009 --text_lang "en" --lang "zh"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB1891 --text_lang "en" --lang "zh"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0005 --text_lang "en" --lang "zh"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0603 --text_lang "en" --lang "zh"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0710 --text_lang "en" --lang "zh"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0966 --text_lang "en" --lang "zh"

python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0009 --text_lang "en" --lang "en"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB1891 --text_lang "en" --lang "en"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0005 --text_lang "en" --lang "en"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0603 --text_lang "en" --lang "en"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0710 --text_lang "en" --lang "en"
python3 synthesize.py --text "we had to send to the english hospital and borrow some." --restore_step 100000 --mode single --dataset CrossLingual --speaker_id SSB0966 --text_lang "en" --lang "en"


# synth en spker: 1731 3615 1382 / 4535 2299 922

python3 synthesize.py --text "the poetry of nature, which nothing can destroy" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 1731 --text_lang "en" --lang "en"
python3 synthesize.py --text "the poetry of nature, which nothing can destroy" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 3615 --text_lang "en" --lang "en"
python3 synthesize.py --text "the poetry of nature, which nothing can destroy" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 1382 --text_lang "en" --lang "en"

python3 synthesize.py --text "when he was in, he shouted at the top of his voice" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 4535 --text_lang "en" --lang "en"
python3 synthesize.py --text "when he was in, he shouted at the top of his voice" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 2299 --text_lang "en" --lang "en"
python3 synthesize.py --text "when he was in, he shouted at the top of his voice" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 922 --text_lang "en" --lang "en"

python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 1731 --text_lang "zh" --lang "en"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 3615 --text_lang "zh" --lang "en"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 1382 --text_lang "zh" --lang "en"

python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 4535 --text_lang "zh" --lang "en"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 2299 --text_lang "zh" --lang "en"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 922 --text_lang "zh" --lang "en"

python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 1731 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 3615 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 1382 --text_lang "zh" --lang "zh"

python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 4535 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 2299 --text_lang "zh" --lang "zh"
python3 synthesize.py --text "五萬四千三百二十一" --restore_step 100000 --mode single --dataset CrossLingual --speaker_id 922 --text_lang "zh" --lang "zh"
