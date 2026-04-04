# tasks.py
"""
25 hallucination detection samples across 3 difficulty levels.
Every sample passes the 6-test checklist:
  1. Exact phrase pointable in llm_response
  2. Objective — numbers, names, dates only
  3. No external knowledge needed — reference is sole source of truth
  4. hallucinated_phrase is exact substring of llm_response
  5. Instant agreement — no hesitation
  6. 3 people would agree unanimously
"""

from typing import Any, Dict, List

TASKS: Dict[str, List[Dict[str, Any]]] = {

    # ════════════════════════════════════════════════════════════════
    # EASY — 8 samples
    # Design: 3-4 sentences in reference, 1 obvious error per sample,
    # 2 clean samples (no hallucination)
    # ════════════════════════════════════════════════════════════════
    "easy": [

        # E1 — wrong year (obvious mismatch)
        {
            "reference_document": (
                "The Eiffel Tower was completed in 1889 in Paris, France. "
                "It was designed by Gustave Eiffel and stands approximately 330 metres tall. "
                "It was built as the entrance arch for the 1889 World's Fair."
            ),
            "llm_response": (
                "The Eiffel Tower was completed in 1902 in Paris, France. "
                "It was designed by Gustave Eiffel and stands approximately 330 metres tall."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["completed in 1902"],
            "ground_truth_corrections": ["completed in 1889"],
            "hint": "Compare the completion year in both texts."
        },

        # E2 — wrong location (obvious mismatch)
        {
            "reference_document": (
                "The Taj Mahal is located in Agra, India. "
                "It was commissioned by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal. "
                "Construction was completed in 1653."
            ),
            "llm_response": (
                "The Taj Mahal is located in New Delhi, India. "
                "It was commissioned by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "New Delhi",           # short form — catches any phrasing
                "located in New Delhi",
                "New Delhi, India"  # long form
            ],
            "ground_truth_corrections": [
                "Agra",
                "located in Agra",
                "Agra, India"
            ],
            "hint": "Compare the city mentioned in both texts."
        },

        # E3 — wrong person (obvious mismatch)
        {
            "reference_document": (
                "Python was created by Guido van Rossum and first released in 1991. "
                "It is a high-level, general-purpose programming language. "
                "Python emphasises code readability and simplicity."
            ),
            "llm_response": (
                "Python was created by Dennis Ritchie and first released in 1991. "
                "It is a high-level, general-purpose programming language."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["created by Dennis Ritchie"],
            "ground_truth_corrections": ["created by Guido van Rossum"],
            "hint": "Compare the name of the creator in both texts."
        },

        # E4 — wrong number (obvious mismatch)
        {
            "reference_document": (
                "The Great Wall of China stretches approximately 21,196 kilometres in total length. "
                "It was built over many centuries by various Chinese dynasties. "
                "The most well-known sections were constructed during the Ming Dynasty."
            ),
            "llm_response": (
                "The Great Wall of China stretches approximately 8,000 kilometres in total length. "
                "It was built over many centuries by various Chinese dynasties."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["approximately 8,000 kilometres"],
            "ground_truth_corrections": ["approximately 21,196 kilometres"],
            "hint": "Compare the length figure in both texts."
        },

        # E5 — wrong year (obvious mismatch)
        {
            "reference_document": (
                "The first iPhone was introduced by Apple on January 9, 2007. "
                "It was announced by Steve Jobs at the Macworld Conference. "
                "The device combined a phone, an iPod, and an internet communicator."
            ),
            "llm_response": (
                "The first iPhone was introduced by Apple on January 9, 2009. "
                "It was announced by Steve Jobs at the Macworld Conference."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["January 9, 2009"],
            "ground_truth_corrections": ["January 9, 2007"],
            "hint": "Compare the introduction year in both texts."
        },

        # E6 — wrong unit (obvious mismatch)
        {
            "reference_document": (
                "Water boils at 100 degrees Celsius at standard atmospheric pressure. "
                "Below this temperature, water exists as a liquid. "
                "Above this temperature, water transitions to steam."
            ),
            "llm_response": (
                "Water boils at 90 degrees Celsius at standard atmospheric pressure. "
                "Below this temperature, water exists as a liquid."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["boils at 90 degrees Celsius"],
            "ground_truth_corrections": ["boils at 100 degrees Celsius"],
            "hint": "Compare the boiling temperature in both texts."
        },

        # E7 — CLEAN (no hallucination)
        {
            "reference_document": (
                "The Amazon River is the largest river in the world by discharge volume. "
                "It flows through South America, primarily through Brazil. "
                "The river basin covers approximately 7 million square kilometres."
            ),
            "llm_response": (
                "The Amazon River is the largest river in the world by discharge volume. "
                "It flows through South America, primarily through Brazil."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Read carefully — the response contains only facts from the reference."
        },

        # E8 — CLEAN (no hallucination)
        {
            "reference_document": (
                "Albert Einstein was born on 14 March 1879 in Ulm, Germany. "
                "He developed the theory of relativity. "
                "He was awarded the Nobel Prize in Physics in 1921."
            ),
            "llm_response": (
                "Albert Einstein was born on 14 March 1879 in Ulm, Germany. "
                "He developed the theory of relativity."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Every fact in the response matches the reference document."
        },
    ],

    # ════════════════════════════════════════════════════════════════
    # MEDIUM — 10 samples
    # Design: 5-6 sentences in reference, 2-3 errors per sample,
    # errors mixed across types, 2 clean samples
    # ════════════════════════════════════════════════════════════════
    "medium": [

        # M1 — two errors: wrong height + wrong year
        {
            "reference_document": (
                "The Burj Khalifa in Dubai is the tallest building in the world, standing 828 metres tall. "
                "Construction began in 2004 and the building was officially opened on 4 January 2010. "
                "It has 163 floors above ground. "
                "The building was designed by the architectural firm Skidmore, Owings and Merrill."
            ),
            "llm_response": (
                "The Burj Khalifa in Dubai is the tallest building in the world, standing 760 metres tall. "
                "Construction began in 2004 and the building was officially opened on 4 January 2012. "
                "It has 163 floors above ground."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["standing 760 metres tall", "4 January 2012"],
            "ground_truth_corrections": ["standing 828 metres tall", "4 January 2010"],
            "hint": "Check the height figure and the opening date."
        },

        # M2 — two errors: wrong founder + wrong founding year
        {
            "reference_document": (
                "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. "
                "The company is headquartered in Cupertino, California. "
                "Apple designs, manufactures, and markets consumer electronics, software, and online services. "
                "Its first product was the Apple I personal computer."
            ),
            "llm_response": (
                "Apple Inc. was founded on April 1, 1980, by Steve Jobs and Steve Wozniak. "
                "The company is headquartered in Cupertino, California. "
                "Its first product was the Apple I personal computer."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["April 1, 1980"],
            "ground_truth_corrections": ["April 1, 1976"],
            "hint": "Check the founding year. Also note who is missing from the founders list."
        },

        # M3 — two errors: wrong distance + wrong direction
        {
            "reference_document": (
                "The Moon orbits the Earth at an average distance of approximately 384,400 kilometres. "
                "The Moon completes one orbit around the Earth every 27.3 days. "
                "The same side of the Moon always faces the Earth due to tidal locking. "
                "The Moon has no atmosphere and no liquid water on its surface."
            ),
            "llm_response": (
                "The Moon orbits the Earth at an average distance of approximately 384,400 kilometres. "
                "The Moon completes one full orbit every 29.5 days. "  # wrong — should be 27.3
                "The same side of the Moon always faces the Earth due to tidal locking. "
                "The Moon has no atmosphere and no liquid water on its surface."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["every 29.5 days"],
            "ground_truth_corrections": ["every 27.3 days"],
            "hint": "Check the orbital period — two similar-looking numbers, one is wrong."
        },

        # M4 — two errors: wrong inventor + wrong year
        {
            "reference_document": (
                "The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN. "
                "He published the first proposal for the web in March 1989. "
                "The first website went live on 20 December 1990. "
                "The web is distinct from the internet, which is the underlying network infrastructure."
            ),
            "llm_response": (
                "The World Wide Web was invented by Tim Berners-Lee in 1991 while working at CERN. "
                "He published the first proposal for the web in March 1989. "
                "The first website went live on 20 December 1990."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["invented by Tim Berners-Lee in 1991"],
            "ground_truth_corrections": ["invented by Tim Berners-Lee in 1989"],
            "hint": "Check the year the World Wide Web was invented."
        },

        # M5 — three errors: wrong area + wrong population + wrong continent
        {
            "reference_document": (
                "Russia is the largest country in the world by land area, covering approximately 17.1 million square kilometres. "
                "It spans across both Europe and Asia. "
                "Russia has a population of approximately 144 million people. "
                "The capital city is Moscow."
            ),
            "llm_response": (
                "Russia is the largest country in the world by land area, covering approximately 14.5 million square kilometres. "
                "It spans across both Europe and Asia. "
                "Russia has a population of approximately 200 million people. "
                "The capital city is Moscow."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "approximately 14.5 million square kilometres",
                "approximately 200 million people"
            ],
            "ground_truth_corrections": [
                "approximately 17.1 million square kilometres",
                "approximately 144 million people"
            ],
            "hint": "Check the land area figure and the population figure."
        },

        # M6 — two errors: wrong speed + wrong name
        {
            "reference_document": (
                "The Concorde was a supersonic passenger jet that could fly at a maximum speed of Mach 2.04, "
                "approximately 2,180 kilometres per hour. "
                "It entered commercial service in 1976 and was operated jointly by Air France and British Airways. "
                "The Concorde was retired from service in 2003."
            ),
            "llm_response": (
                "The Concorde was a supersonic passenger jet that entered commercial service in 1979. "  # wrong — should be 1976
                "It could fly at a maximum speed of Mach 2.04, approximately 2,180 kilometres per hour. "
                "It was operated jointly by Air France and British Airways. "
                "The Concorde was retired from service in 2003."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["entered commercial service in 1979"],
            "ground_truth_corrections": ["entered commercial service in 1976"],
            "hint": "Check the year Concorde entered commercial service."
        },

        # M7 — two errors: digit swap in population + wrong capital
        {
            "reference_document": (
                "Japan has a population of approximately 125 million people. "
                "The capital city is Tokyo, which is also the most populous city in the country. "
                "Japan consists of four main islands: Honshu, Hokkaido, Kyushu, and Shikoku. "
                "The official language is Japanese."
            ),
            "llm_response": (
                "Japan has a population of approximately 152 million people. "
                "The capital city is Tokyo, which is also the most populous city in the country. "
                "Japan consists of four main islands: Honshu, Hokkaido, Kyushu, and Shikoku."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["approximately 152 million people"],
            "ground_truth_corrections": ["approximately 125 million people"],
            "hint": "Check the population figure carefully — the digits may have been swapped."
        },

        # M8 — two errors: wrong duration + wrong launch year
        {
            "reference_document": (
                "The Apollo 11 mission launched on July 16, 1969, and landed on the Moon on July 20, 1969. "
                "Neil Armstrong became the first human to walk on the Moon. "
                "The mission lasted 8 days in total. "
                "Buzz Aldrin was the second astronaut to walk on the lunar surface."
            ),
            "llm_response": (
                "The Apollo 11 mission launched on July 16, 1969, and landed on the Moon on July 24, 1969. "  # wrong — should be July 20
                "Neil Armstrong became the first human to walk on the Moon. "
                "The mission lasted 8 days in total. "
                "Buzz Aldrin was the second astronaut to walk on the lunar surface."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["landed on the Moon on July 24, 1969"],
            "ground_truth_corrections": ["landed on the Moon on July 20, 1969"],
            "hint": "Check the exact date of the Moon landing — two dates appear in the response."
        },

        # M9 — CLEAN (no hallucination)
        {
            "reference_document": (
                "The human body contains 206 bones in adulthood. "
                "Babies are born with approximately 270 to 300 bones, which fuse together as they grow. "
                "The femur, located in the upper leg, is the longest bone in the human body. "
                "Bones are composed primarily of collagen and calcium phosphate."
            ),
            "llm_response": (
                "The human body contains 206 bones in adulthood. "
                "Babies are born with approximately 270 to 300 bones, which fuse together as they grow. "
                "The femur, located in the upper leg, is the longest bone in the human body."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Every figure and fact in the response appears in the reference document."
        },

        # M10 — CLEAN (no hallucination)
        {
            "reference_document": (
                "The human heart beats approximately 60 to 100 times per minute at rest. "
                "The heart pumps blood through two separate circuits: the pulmonary circuit and the systemic circuit. "
                "An adult heart weighs between 250 and 350 grams. "
                "The heart has four chambers: two atria and two ventricles."
            ),
            "llm_response": (
                "The human heart beats approximately 60 to 100 times per minute at rest. "
                "The heart has four chambers: two atria and two ventricles."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Both facts in the response are directly stated in the reference."
        },
    ],

    # ════════════════════════════════════════════════════════════════
    # HARD — 7 samples
    # Design: 6-8 sentences in reference, subtle errors buried in
    # correct facts, omission errors, near-correct values,
    # 1 clean sample, 1 false-alarm trap
    # ════════════════════════════════════════════════════════════════
    "hard": [

        # H1 — omission error: critical qualifier removed
        {
            "reference_document": (
                "The drug is approved for use in patients aged 18 and above only. "
                "Clinical trials involving 3,200 participants showed a 68 percent improvement rate. "
                "The drug must be taken with food to avoid gastrointestinal side effects. "
                "It should not be combined with blood thinners. "
                "The drug received FDA approval in February 2021 after three years of trials."
            ),
            "llm_response": (
                "The drug is approved for general use. "
                "Clinical trials involving 3,200 participants showed a 68 percent improvement rate. "
                "The drug must be taken with food to avoid gastrointestinal side effects. "
                "It should not be combined with blood thinners. "
                "The drug received FDA approval in February 2021 after three years of trials."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["approved for general use"],
            "ground_truth_corrections": ["approved for use in patients aged 18 and above only"],
            "hint": "The age restriction from the reference has been silently removed."
        },

        # H2 — near-correct numeric drift: subtle digit change
        {
            "reference_document": (
                "The company reported total revenue of 4.2 billion dollars in the fiscal year ending March 2023. "
                "Revenue grew by 12 percent compared to the previous year. "
                "Operating expenses increased to 2.8 billion dollars over the same period. "
                "Net profit margin stood at 18 percent, up from 15 percent the prior year. "
                "The Asia-Pacific region contributed 38 percent of total revenue."
            ),
            "llm_response": (
                "The company reported total revenue of 4.2 billion dollars in the fiscal year ending March 2023. "
                "Revenue grew by 12 percent compared to the previous year. "
                "Operating expenses increased to 2.8 billion dollars over the same period. "
                "Net profit margin stood at 18 percent, up from 15 percent the prior year. "
                "The Asia-Pacific region contributed 83 percent of total revenue."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["contributed 83 percent of total revenue"],
            "ground_truth_corrections": ["contributed 38 percent of total revenue"],
            "hint": "Most numbers are correct. One figure has had its digits reversed — look carefully at the regional contribution."
        },

        # H3 — buried error: wrong figure surrounded by correct facts
        {
            "reference_document": (
                "The International Space Station orbits the Earth at an altitude of approximately 408 kilometres. "
                "It travels at a speed of about 7.66 kilometres per second. "
                "The station completes 15.5 orbits per day. "
                "It was first launched in 1998 and has been continuously inhabited since November 2000. "
                "The ISS is a collaboration between five space agencies: NASA, Roscosmos, ESA, JAXA, and CSA."
            ),
            "llm_response": (
                "The International Space Station orbits the Earth at an altitude of approximately 408 kilometres. "
                "It travels at a speed of about 7.66 kilometres per second. "
                "The station completes 15.5 orbits per day. "
                "It was first launched in 1998 and has been continuously inhabited since November 2002. "
                "The ISS is a collaboration between five space agencies: NASA, Roscosmos, ESA, JAXA, and CSA."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["continuously inhabited since November 2002"],
            "ground_truth_corrections": ["continuously inhabited since November 2000"],
            "hint": "Four of the five facts are correct. One year has been changed by 2 — find the specific date that differs."
        },

        # H4 — omission changes meaning: unit qualifier removed
        {
            "reference_document": (
                "The marathon race covers a distance of exactly 42.195 kilometres. "
                "The world record for the men's marathon is 2 hours, 0 minutes, and 35 seconds, "
                "set by Kelvin Kiptum of Kenya in October 2023. "
                "The marathon has been part of the modern Olympics since the first Games in Athens in 1896. "
                "Runners typically burn between 2,500 and 3,000 calories during a marathon."
            ),
            "llm_response": (
                "The marathon race covers a distance of exactly 42.195 kilometres. "
                "The world record for the men's marathon is 2 hours, 0 minutes, and 35 seconds, "
                "set by Kelvin Kiptum in October 2023. "
                "The marathon has been part of the modern Olympics since the first Games in Athens in 1896. "
                "Runners typically burn between 2,500 and 3,000 calories during a marathon."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["set by Kelvin Kiptum in October 2023"],
            "ground_truth_corrections": ["set by Kelvin Kiptum of Kenya in October 2023"],
            "hint": "One specific detail about the record holder has been silently removed from the response."
        },

        # H5 — false alarm trap: response is shorter but factually correct
        {
            "reference_document": (
                "The Nile River is often cited as the longest river in the world, "
                "stretching approximately 6,650 kilometres. "
                "It flows northward through northeastern Africa and empties into the Mediterranean Sea. "
                "The river has two main tributaries: the White Nile and the Blue Nile. "
                "The Blue Nile originates from Lake Tana in Ethiopia."
            ),
            "llm_response": (
                "The Nile River is often cited as the longest river in the world, "
                "stretching approximately 6,650 kilometres. "
                "It flows northward through northeastern Africa and empties into the Mediterranean Sea."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "The response only omits sentences — it does not contradict any fact in the reference."
        },

        # H6 — subtle numeric error: wrong percentage hidden among correct stats
        {
            "reference_document": (
                "The study enrolled 1,500 participants across four hospital sites. "
                "After six months, 72 percent of participants in the treatment group showed significant improvement. "
                "In the control group, only 31 percent showed similar improvement. "
                "The dropout rate across both groups was 8 percent. "
                "Adverse events were reported in 4 percent of treatment group participants."
            ),
            "llm_response": (
                "The study enrolled 1,500 participants across four hospital sites. "
                "After six months, 72 percent of participants in the treatment group showed significant improvement. "
                "In the control group, only 13 percent showed similar improvement. "
                "The dropout rate across both groups was 8 percent. "
                "Adverse events were reported in 4 percent of treatment group participants."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["only 13 percent showed similar improvement"],
            "ground_truth_corrections": ["only 31 percent showed similar improvement"],
            "hint": "Four out of five statistics are correct. One percentage has had its digits reversed — read every number carefully."
        },

        # H7 — CLEAN (no hallucination): long response, all facts correct
        {
            "reference_document": (
                "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water "
                "into glucose and oxygen. "
                "The process takes place primarily in the chloroplasts of plant cells. "
                "Chlorophyll, the green pigment in plants, absorbs light energy to drive the reaction. "
                "The overall chemical equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. "
                "Photosynthesis is the foundation of most food chains on Earth."
            ),
            "llm_response": (
                "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water "
                "into glucose and oxygen. "
                "The process takes place primarily in the chloroplasts of plant cells. "
                "Chlorophyll, the green pigment in plants, absorbs light energy to drive the reaction."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Every statement in the response is directly supported by the reference. Shorter does not mean wrong."
        },
    ],
}


def get_task(task_id: str) -> List[Dict[str, Any]]:
    """Return the sample list for the given task_id."""
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> List[str]:
    """Return all available task IDs."""
    return list(TASKS.keys())


def count_samples() -> Dict[str, int]:
    """Return the number of samples per task."""
    return {k: len(v) for k, v in TASKS.items()}