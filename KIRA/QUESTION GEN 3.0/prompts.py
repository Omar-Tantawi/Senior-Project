"""
All prompts in English and Arabic.
The LLM receives the prompt in the same language as the document.
Includes few-shot examples to guide the model toward high-quality output.
"""

# ── Bloom's taxonomy descriptions ────────────────────────────────────────────

BLOOM_EN = {
    "remember":  "Recall facts and basic concepts (e.g., Define, List, Name, Who, What, When).",
    "understand":"Explain ideas or concepts in your own words (e.g., Summarize, Describe, Explain, Classify).",
    "apply":     "Use information in new situations (e.g., Calculate, Solve, Use, Demonstrate).",
    "analyze":   "Draw connections and break down information (e.g., Compare, Contrast, Examine, Differentiate).",
    "evaluate":  "Justify a decision or course of action (e.g., Argue, Judge, Defend, Justify, Critique).",
    "create":    "Produce new or original work (e.g., Design, Construct, Develop, Formulate, Propose).",
}

BLOOM_AR = {
    "remember":  "تذكّر الحقائق والمفاهيم الأساسية (مثل: عرّف، اذكر، سمّ، من، ما، متى).",
    "understand":"شرح الأفكار بأسلوبك الخاص (مثل: لخّص، اوصف، اشرح، صنّف).",
    "apply":     "توظيف المعلومات في مواقف جديدة (مثل: احسب، حلّ، طبّق، بيّن).",
    "analyze":   "تحليل المعلومات وربطها (مثل: قارن، فرّق، افحص، حلّل).",
    "evaluate":  "تقييم وإصدار أحكام مبررة (مثل: ناقش، قيّم، دافع، انتقد).",
    "create":    "إنتاج أعمال أو أفكار جديدة (مثل: صمّم، ابنِ، طوّر، اقترح).",
}

# ── Question type instructions ────────────────────────────────────────────────

TYPE_INSTRUCTIONS_EN = {
    "mcq": """Generate a Multiple Choice Question (MCQ):
- One clear question
- 4 options labeled A, B, C, D
- Exactly one correct answer
- 3 plausible but clearly wrong distractors
- Provide the correct answer letter and a brief explanation""",

    "true_false": """Generate a True/False question:
- One clear statement that is definitively true or false
- Provide the correct answer (True or False)
- Provide a brief explanation""",

    "short_answer": """Generate a Short Answer question:
- One clear question requiring a 1-3 sentence answer
- Provide a model answer""",

    "fill_blank": """Generate a Fill in the Blank question:
- One sentence with one key word/phrase replaced by _______
- The blank must be a meaningful, specific term from the text
- Provide the correct answer""",

    "essay": """Generate an Essay question:
- One open-ended question requiring a detailed paragraph or more
- Provide key points that a good answer should cover""",
}

TYPE_INSTRUCTIONS_AR = {
    "mcq": """أنشئ سؤال اختيار من متعدد (MCQ):
- سؤال واضح ومحدد
- 4 خيارات مسماة بـ A وB وC وD
- إجابة صحيحة واحدة فقط
- 3 خيارات خاطئة ولكن معقولة
- اذكر حرف الإجابة الصحيحة مع شرح مختصر""",

    "true_false": """أنشئ سؤال صح أم خطأ:
- جملة واضحة تكون إما صحيحة أو خاطئة بشكل قاطع
- اذكر الإجابة (صح أو خطأ) مع شرح مختصر""",

    "short_answer": """أنشئ سؤال إجابة قصيرة:
- سؤال واضح يتطلب إجابة من 1-3 جمل
- قدّم نموذج إجابة""",

    "fill_blank": """أنشئ سؤال ملء الفراغ:
- جملة واحدة تحتوي على فراغ _______ يمثل مصطلحاً أساسياً
- اذكر الإجابة الصحيحة""",

    "essay": """أنشئ سؤال مقالي:
- سؤال مفتوح يتطلب إجابة تفصيلية
- اذكر النقاط الرئيسية التي يجب أن تتضمنها الإجابة الجيدة""",
}

# ── Difficulty ────────────────────────────────────────────────────────────────

DIFFICULTY_EN = {
    "easy":   "The question should be straightforward and suitable for basic recall.",
    "medium": "The question should require some thinking and understanding.",
    "hard":   "The question should be challenging and require deep understanding or analysis.",
}

DIFFICULTY_AR = {
    "easy":   "يجب أن يكون السؤال مباشراً ومناسباً للتذكر الأساسي.",
    "medium": "يجب أن يتطلب السؤال بعض التفكير والفهم.",
    "hard":   "يجب أن يكون السؤال صعباً ويتطلب فهماً عميقاً أو تحليلاً.",
}

# ── JSON output format instructions ──────────────────────────────────────────

JSON_FORMAT_EN = {
    "mcq": """{
  "type": "mcq",
  "bloom_level": "<level>",
  "difficulty": "<difficulty>",
  "question": "<question text>",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "<A|B|C|D>",
  "explanation": "<why this answer is correct>"
}""",

    "true_false": """{
  "type": "true_false",
  "bloom_level": "<level>",
  "difficulty": "<difficulty>",
  "question": "<statement>",
  "correct_answer": "<True|False>",
  "explanation": "<explanation>"
}""",

    "short_answer": """{
  "type": "short_answer",
  "bloom_level": "<level>",
  "difficulty": "<difficulty>",
  "question": "<question>",
  "model_answer": "<expected answer>"
}""",

    "fill_blank": """{
  "type": "fill_blank",
  "bloom_level": "<level>",
  "difficulty": "<difficulty>",
  "question": "<sentence with _______>",
  "correct_answer": "<the missing word/phrase>"
}""",

    "essay": """{
  "type": "essay",
  "bloom_level": "<level>",
  "difficulty": "<difficulty>",
  "question": "<essay question>",
  "key_points": ["point 1", "point 2", "point 3"]
}""",
}

# Arabic format uses same structure, LLM fills content in Arabic
JSON_FORMAT_AR = JSON_FORMAT_EN


# ── Few-shot examples ─────────────────────────────────────────────────────────
# One perfect example per question type in each language.
# Shown to the model BEFORE asking it to generate, so it learns the expected format.

FEW_SHOT_EN = {
    "mcq": """{
  "type": "mcq",
  "bloom_level": "remember",
  "difficulty": "medium",
  "question": "What is the main function of the mitochondria in a cell?",
  "options": {
    "A": "To store genetic information",
    "B": "To produce energy in the form of ATP",
    "C": "To synthesize proteins",
    "D": "To control what enters and exits the cell"
  },
  "correct_answer": "B",
  "explanation": "Mitochondria are known as the powerhouse of the cell because they generate most of the cell's supply of ATP through cellular respiration."
}""",

    "true_false": """{
  "type": "true_false",
  "bloom_level": "understand",
  "difficulty": "easy",
  "question": "The speed of light in a vacuum is approximately 300,000 kilometers per second.",
  "correct_answer": "True",
  "explanation": "The speed of light in a vacuum is exactly 299,792,458 m/s, commonly rounded to 300,000 km/s."
}""",

    "short_answer": """{
  "type": "short_answer",
  "bloom_level": "understand",
  "difficulty": "medium",
  "question": "Explain why plants appear green to the human eye.",
  "model_answer": "Plants appear green because their leaves contain chlorophyll, a pigment that absorbs red and blue light but reflects green light. The reflected green light is what reaches our eyes, making the plant appear green."
}""",

    "fill_blank": """{
  "type": "fill_blank",
  "bloom_level": "remember",
  "difficulty": "easy",
  "question": "The process by which plants make their own food using sunlight is called _______.",
  "correct_answer": "photosynthesis"
}""",

    "essay": """{
  "type": "essay",
  "bloom_level": "analyze",
  "difficulty": "hard",
  "question": "Analyze the role of the water cycle in maintaining life on Earth. How would disruptions to this cycle affect ecosystems?",
  "key_points": [
    "Definition and stages of the water cycle (evaporation, condensation, precipitation)",
    "How the water cycle distributes fresh water across the planet",
    "Impact on plant life, agriculture, and drinking water availability",
    "Examples of disruptions: drought, flooding, climate change effects"
  ]
}""",
}

FEW_SHOT_AR = {
    "mcq": """{
  "type": "mcq",
  "bloom_level": "remember",
  "difficulty": "medium",
  "question": "ما الوظيفة الرئيسية للميتوكوندريا في الخلية؟",
  "options": {
    "A": "تخزين المعلومات الوراثية",
    "B": "إنتاج الطاقة على شكل ATP",
    "C": "تصنيع البروتينات",
    "D": "التحكم فيما يدخل الخلية ويخرج منها"
  },
  "correct_answer": "B",
  "explanation": "تُعرف الميتوكوندريا بأنها محطة توليد الطاقة في الخلية، إذ تنتج معظم الطاقة اللازمة للخلية في صورة ATP عن طريق التنفس الخلوي."
}""",

    "true_false": """{
  "type": "true_false",
  "bloom_level": "understand",
  "difficulty": "easy",
  "question": "تبلغ سرعة الضوء في الفراغ نحو 300,000 كيلومتر في الثانية.",
  "correct_answer": "True",
  "explanation": "سرعة الضوء في الفراغ تساوي تقريباً 299,792,458 متراً في الثانية، وتُقرَّب إلى 300,000 كيلومتر في الثانية."
}""",

    "short_answer": """{
  "type": "short_answer",
  "bloom_level": "understand",
  "difficulty": "medium",
  "question": "اشرح لماذا تبدو النباتات خضراء اللون للعين البشرية.",
  "model_answer": "تبدو النباتات خضراء لأن أوراقها تحتوي على الكلوروفيل، وهو صبغة تمتص الضوء الأحمر والأزرق وتعكس الضوء الأخضر. هذا الضوء الأخضر المنعكس هو ما يصل إلى أعيننا، مما يجعل النبات يبدو أخضر."
}""",

    "fill_blank": """{
  "type": "fill_blank",
  "bloom_level": "remember",
  "difficulty": "easy",
  "question": "العملية التي تصنع فيها النباتات غذاءها باستخدام ضوء الشمس تسمى _______.",
  "correct_answer": "البناء الضوئي"
}""",

    "essay": """{
  "type": "essay",
  "bloom_level": "analyze",
  "difficulty": "hard",
  "question": "حلّل دور دورة الماء في الحفاظ على الحياة على الأرض. كيف تؤثر الاضطرابات في هذه الدورة على النظم البيئية؟",
  "key_points": [
    "تعريف مراحل دورة الماء (التبخر، التكثف، التساقط)",
    "كيف تعيد دورة الماء توزيع المياه العذبة عبر الكوكب",
    "التأثير على الحياة النباتية والزراعة ومصادر مياه الشرب",
    "أمثلة على الاضطرابات: الجفاف، الفيضانات، تأثيرات تغير المناخ"
  ]
}""",
}


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt(
    context:       str,
    question_type: str,
    bloom_level:   str,
    difficulty:    str,
    language:      str,          # "ar" or "en"
    question_num:  int = 1,
    topic:         str = "",     # optional focus topic for this question
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for the LLM.

    Grounding-first: the model is told to use ONLY the reference passage and no
    outside knowledge. We deliberately DO NOT include a worked topical few-shot
    example — on a small model those examples leak their subject (mitochondria,
    photosynthesis…) into unrelated documents. The JSON format skeleton below is
    enough to teach the output shape.
    """
    is_arabic = language == "ar"

    bloom_desc  = (BLOOM_AR if is_arabic else BLOOM_EN).get(bloom_level, "")
    type_instr  = (TYPE_INSTRUCTIONS_AR if is_arabic else TYPE_INSTRUCTIONS_EN).get(question_type, "")
    diff_desc   = (DIFFICULTY_AR if is_arabic else DIFFICULTY_EN).get(difficulty, "")
    json_format = (JSON_FORMAT_AR if is_arabic else JSON_FORMAT_EN).get(question_type, "")

    if is_arabic:
        system = (
            "أنت معلّم خبير تكتب أسئلة امتحانية عالية الجودة من نص مدرسي. "
            "اعتمد حصراً على النص المرجعي المعطى ولا تستخدم أي معلومات من خارجه. "
            "CRITICAL: أجب بصيغة JSON صحيحة فقط — بدون أي نص إضافي أو علامات markdown. "
            "CRITICAL: جميع مفاتيح JSON بالإنجليزية تماماً كما في القالب. "
            "محتوى السؤال والإجابات والشرح يجب أن يكون بالعربية فقط."
        )
        topic_line = f"ركّز السؤال على هذا الموضوع الوارد في النص: «{topic}».\n" if topic else ""
        user = f"""النص المرجعي:
\"\"\"
{context}
\"\"\"

{topic_line}المطلوب: أنشئ سؤالاً رقم {question_num} بالمواصفات التالية:

نوع السؤال: {type_instr}

مستوى بلوم: {bloom_level} — {bloom_desc}

الصعوبة: {difficulty} — {diff_desc}

قواعد صارمة:
- يجب أن يستند السؤال وإجابته بشكل كامل إلى النص المرجعي أعلاه فقط.
- لا تخترع أي معلومة غير موجودة في النص. إن لم يكفِ النص لهذا النوع، اعتمد أهم معلومة واردة فيه.
- اكتب نص السؤال والخيارات والشرح بالعربية، واستخدم مفاتيح JSON الإنجليزية فقط.

أجب بصيغة JSON فقط، بهذا القالب الدقيق وبمفاتيح إنجليزية:
{json_format}"""

    else:
        system = (
            "You are an expert educator who writes high-quality exam questions from a "
            "school textbook passage. Use ONLY the provided reference text — never outside "
            "knowledge. You follow Bloom's Taxonomy. "
            "Always respond with valid JSON only — no extra text, no markdown code blocks. "
            "All JSON keys must be in English."
        )
        topic_line = f"Focus the question on this topic from the text: \"{topic}\".\n" if topic else ""
        user = f"""Reference text:
\"\"\"
{context}
\"\"\"

{topic_line}Task: Generate question number {question_num} with the following specifications:

Question type: {type_instr}

Bloom's level: {bloom_level} — {bloom_desc}

Difficulty: {difficulty} — {diff_desc}

Strict rules:
- The question AND its answer must be fully supported by the reference text above — only.
- Do not invent information not in the text. If the text can't support this type, base the
  question on the most important fact present in the text.

Respond with JSON only, in this exact format:
{json_format}"""

    return system, user


def build_outline_prompt(samples: list[str], n: int, language: str) -> tuple[str, str]:
    """
    Ask the model for the real topics covered in a book, given sampled excerpts.
    Returns (system, user). The reply is expected to be a JSON array of short phrases;
    these are used as retrieval queries to spread questions across actual topics.
    """
    is_arabic = language == "ar"
    joined = "\n- ".join(samples)
    target = max(n, 6)

    if is_arabic:
        system = (
            "أنت تحلل مقتطفات من كتاب مدرسي وتستخرج المواضيع الفعلية التي يغطيها. "
            "أجب بمصفوفة JSON من عبارات قصيرة فقط، دون أي نص آخر."
        )
        user = (
            f"مقتطفات من الكتاب:\n- {joined}\n\n"
            f"اذكر حتى {target} من المواضيع أو المفاهيم الرئيسية التي وردت فعلاً في المقتطفات أعلاه. "
            f"أعد فقط مصفوفة JSON من عبارات قصيرة بالعربية، مثل: [\"الموضوع الأول\", \"الموضوع الثاني\"]."
        )
    else:
        system = (
            "You extract the concrete topics actually covered in textbook excerpts. "
            "Respond with a JSON array of short topic phrases only — no other text."
        )
        user = (
            f"Textbook excerpts:\n- {joined}\n\n"
            f"List up to {target} key topics/concepts that actually appear in the excerpts above. "
            f"Return ONLY a JSON array of short phrases, e.g. [\"first topic\", \"second topic\"]."
        )
    return system, user
