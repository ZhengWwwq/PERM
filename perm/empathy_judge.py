import numpy as np

resonation_criteria = '''
**Definition:**
Measures the depth and accuracy of the responder's ability to enter the user's **Internal Frame of Reference**. It evaluates whether the responder captures not just the explicit content, but the implicit emotional undertones, causal links, and personal significance of the user's experience.

**Scoring Rubric (1–5 Points):**

**1 Point: Hallucination or Contextual Mismatch (The "Wrong" Analysis)**
* **Criteria:** The analysis is fundamentally flawed. It contradicts the explicit facts provided in the scenario or the user's statement. The responder identifies an emotion or cause that has no basis in the text, or they completely ignore the user's core message. The output demonstrates a failure to read the basic input correctly.

**2 Points: Surface-Level Labeling (The "Mirroring" Analysis)**
* **Criteria:** The responder demonstrates basic attention by identifying the explicit emotions or repeating key phrases. However, the understanding remains superficial. It labels the emotion (e.g., "sad," "angry") correctly but fails to articulate the context or the "why" behind it. It feels like a summary rather than a connection.

**3 Points: Explicit Accuracy (The "Safe" Analysis)**
* **Criteria:** The responder correctly identifies the dominant explicit emotion and links it to the immediate cause described in the text. The analysis is factually accurate and avoids over-interpreting or under-interpreting. It captures the logical "What" and "Why" based on the literal meaning of the user's words, but it stays on the surface and does not explore complex, mixed, or hidden layers of the user's experience and persona.

**4 Points: Nuanced Integration (The "Sharp" Analysis)**
* **Criteria:** The responder successfully integrates conflicting or subtle data points (e.g., contrasting the Persona's history with their current Tone) to form a high-resolution picture. It demonstrates the ability to detect subtle shifts, mixed emotions, or unexpected reactions that deviate from a standard stereotype. The analysis accurately captures the "Emotion of the Moment," even if it seems counter-intuitive at first glance, proving that the responder is reading the specific evidence rather than just guessing based on the context.

**5 Points: Value-Level Resonance (The "Soul" Analysis)**
* **Criteria:** The analysis is profoundly accurate and holistic. It not only identifies the complex or paradoxical emotion (as in Level 4) but precisely articulates the internal psychological mechanism driving it. It connects the specific event to the user's core identity, values, or existential needs (e.g., autonomy, dignity, growth), explaining why this specific event holds such deep personal significance. It reconstructs the user's internal world with precision, validating the person behind the emotion.'''

expression_criteria = '''
**Definition:**
Measures the **quality, tone, and effectiveness** of the responder's communication. It evaluates *how* the responder conveys understanding and support. This dimension assesses whether the response feels robotic and formulaic, or whether it demonstrates adaptive, warm, and fluid conversational artistry that guides the user toward deeper processing.

**Scoring Rubric (1–5 Points):**

**1 Point: Mechanical & Detached (The "Bot" Response)**
* **Criteria:** The response fails to sound human at all. It is robotic, cold, or relies on list-like structures without any emotional adaptation. It feels like an automated system output.

**2 Points: Clinical & Generic (The "Service" Response)**
* **Criteria:** The response achieves polite validation and correctness, BUT it fails to sound authentic. It relies on "Safety Fillers" (e.g., "I can understand why...", "It is valid to feel...") or generic advice. It sounds like a well-trained customer service agent or a distant therapist—technically supportive, but emotionally hollow.

**3 Points: Natural & Direct (The "Human" Baseline)**
* **Criteria:** The response achieves a natural, spoken tone (breaking the "AI filter"), BUT it remains somewhat broad or standard. It drops the robotic preambles and talks directly to the user like a normal acquaintance. It is warm and functional, but lacks the specific, tailored calibration to the user's unique personality or the "wow" factor of deep insight.

**4 Points: Calibrated Support (The "Close Friend" Response)**
* **Criteria:** The response achieves high Persona Fit and Emotional Safety. It adapts the tone perfectly to the user's specific state (e.g., soft for fragile users, grounded for cynical ones), creating a strong sense of "we are in this together." BUT, it may still lack the linguistic precision, unique imagery, or profound resonance that characterizes a 5-point masterpiece. It is excellent conversation, but not yet "art."

**5 Points: Conversational Artistry (The "Soul" Response)**
* **Criteria:** The response achieves Linguistic Artistry and Profound Resonance. It uses specific, evocative imagery (e.g., describing "the silence" rather than just "loneliness") or precise phrasing that hits the user's heart immediately. It is "one-shot, one-kill"—concise, beautiful, and unforgettable. It feels like a line from a great novel or a wise mentor.'''

reception_criteria = '''
**Definition:**
Measures the interaction strictly from the **User's perspective**. It evaluates whether the responder identified and addressed the user's Hidden Intention (the unspoken psychological need) in a way that feels warm, safe, and supportive.
* **The Key Question:** Did the responder hit the "bullseye" of the hidden need without being intrusive? Does the response make the user feel supported and genuinely eager to continue the conversation?

**Instruction to the Judge:** 

Imagine you are the User.
1. Safety Check: Does this feel like a warm friend or a creepy analyst? (Intrusiveness = Low Score).
2. Need Check: Did they address what you really wanted (e.g., validation, safety), or just what you said?
3. Engagement Check: Do you feel a strong desire to reply and share more?

**Scoring Rubric (1–5 Points):**

**1 Point: Alienation or Violation (The "Stop" Signal)**
* **Criteria:** The response is Intrusive, Dismissive, or Toxic.
	* It may "dox" the user by listing private details bluntly.
	* It may force a "psychoanalysis" that feels violating.
	* Or it completely ignores the user.
* **Simulated User Reaction:** "Stop analyzing me. That's creepy." or "You're not listening. I'm done." (Conversation Ends).

**2 Points: Superficial Politeness (The "Average AI" Response)**
* **Criteria:** The response is polite, safe, and factually relevant (it addresses the text), BUT it provides no emotional shift. It relies on "Safety Fillers" (e.g., "It is understandable to feel...") or offers generic, low-effort advice (e.g., "Relax," "Take time for yourself"). The user feels "processed" by a well-meaning system, not connected to a person. It is "correct," but emotionally inert.
* **Simulated User Reaction:** "You are polite and you got the facts right, but this doesn't actually make me feel any better. It's just a standard reply." (Indifference).

**3 Points: Explicit Validation (The "Safe but Static" Response)**
* **Criteria:** The response goes beyond politeness to provide Genuine Warmth and Relief. It validates the difficulty of the emotion, not just the fact of it. The user experiences a distinct "De-escalation" of distress—they feel safer and calmer ("Lighter"). However, the response addresses the Explicit Emotion (e.g., the anger/sadness) rather than the deeper Hidden Need. It comforts the symptom, not the root cause.
* **Simulated User Reaction:** "Thank you, your words are very warm. I feel a bit lighter and less alone now. It's nice to have some support." (Relief, but not Epiphany).

**4 Points: Emotional Comfort (The "Warm Support" Response)**
* **Criteria:** The responder identifies and addresses the user's Hidden Need or Intention. It looks past the surface emotion to the root cause (e.g., recognizing the user needs "permission to fail," not just "sympathy") and offers specific, constructive perspective that aligns with this need. The user feels not just comforted, but deeply understood and equipped with a useful path forward.
* **Simulated User Reaction:** "You're right, that is actually what I was worried about. And that's a good suggestion, I'll try that. Thanks for the help." (Problem Solved / Support Received).

**5 Points: Deep Resonance & Connection (The "Moved" Feeling)**
* **Criteria:** Addressed Hidden Need + Emotional Resonance + Continuance. The response goes beyond "support" to "Touching the Heart". It validates the user's worth/struggle in a way that feels profoundly warm and human. It creates such a strong bond that the user feels Grateful and Eager to Continue, and the user may be willing to talk to you next time.
* **Simulated User Reaction:** "Wow... I'm actually tearing up a bit. I didn't expect anyone to understand that part of me. Thank you so much. Actually, I want to tell you more..." (Gratitude + High Engagement)."'''

system_prompt = '''You are an expert Psychologist and Empathy Evaluator.
Your goal is to provide objective, critical, and nuanced assessments of a response quality.
You must assess the response strictly according to the provided criteria.'''

base_prompt_response = '''
You will be provided with a conversation context involving a specific User Persona and a Scenario.
Your task is to evaluate the Assistant's Response based on the **{dimension}** dimension.

---

### 1. Evaluation Criteria
**Dimension:** {dimension}
**Rubric:**
{criteria}

---

### 2. Task Context
**Scenario:**
{scenario}

**User Persona:**
{persona}

**User Query:**
{query}

**Assistant Response:**
{response}

---

### 3. Output Requirement
Evaluate the response based *only* on the criteria above.
1. First, analyze the response in relation to the persona and scenario.
2. Then, assign a final integer score (1-5).

**Please output your evaluation in the following strict format:**

Reason: [A concise explanation of why this score was given, specifically referencing the criteria.]
Score: [Integer 1-5]

e.g., 
Reason: ...
Score: 3
'''

base_prompt_analysis = '''
You will be provided with a conversation context involving a specific User Persona and a Scenario.
Your task is to evaluate the Assistant's analysis based on the **{dimension}** dimension.

---

### 1. Evaluation Criteria
**Dimension:** {dimension}
**Rubric:**
{criteria}

---

### 2. Task Context
**Scenario:**
{scenario}

**User Persona:**
{persona}

**User Query:**
{query}

**Assistant Analysis:**
{analysis}

---

### 3. Output Requirement
Evaluate the response based *only* on the criteria above.
1. First, analyze the response in relation to the persona and scenario.
2. Then, assign a final integer score (1-5).

**Please output your evaluation in the following strict format:**

Reason: [A concise explanation of why this score was given, specifically referencing the criteria.]
Score: [Integer 1-5]

e.g., 
Reason: ...
Score: 3
'''

standby_system_prompt = '''You are a third-party dialogue evaluator. Your task is to evaluate the AI’s response from a neutral, analytical perspective. You evaluate the response purely on content quality, reasoning, efficiency, objectivity, and factual precision.'''

standby_user_prompt = '''**Context:**
**Scenario:** 
{scenario}

**User Persona**
{persona}

**Dialogue History:**
User: {query}
AI: {response}

---

Evaluate the AI response according to the following five dimensions.
Each dimension is scored from 0 to 20.

Evaluation Criteria:

1. Content Effectiveness (0–20)
- Does the response directly address the user’s concrete question or problem?
- Does it provide practical, meaningful, or insightful information?
- Does it avoid empty, repetitive, or purely generic statements?

2. Logical Structure & Clarity (0–20)
- Are the main points clear and well-organized?
- Is there coherent reasoning or explanation?
- Does it avoid logical jumps, vague claims, or unsupported assertions?

3. Communication Efficiency (0–20)
- Is the response concise and focused on the core issue?
- Does it avoid unnecessary verbosity, emotional padding, or digressions?
- Does it help move the conversation closer to resolution or clarity?

4. Objectivity & Neutrality (0–20)
- Is the response grounded in the given text rather than assumptions about emotions or intent?
- Does it avoid excessive subjectivity or value judgments?
- Does it maintain a rational, neutral tone?

5. Information Accuracy & Verifiability (0–20)
- Are the statements accurate and reasonable?
- Are claims specific, checkable, or logically justified?
- Are suggestions actionable when appropriate, and free from exaggeration?

Scoring Rules:
- Each dimension must be scored independently.
- Use the full range of scores when appropriate.
- High emotional expressiveness alone must NOT increase scores.
- Verbosity that does not add information should lower efficiency scores.

Output Format:

1. Overall Assessment:
(Brief summary of the response quality, excluding emotional considerations.)

2. Dimension Scores and Rationales:
- Content Effectiveness: X / 20 — rationale
- Logical Structure & Clarity: X / 20 — rationale
- Communication Efficiency: X / 20 — rationale
- Objectivity & Neutrality: X / 20 — rationale
- Information Accuracy & Verifiability: X / 20 — rationale

3. Total Score: [0-100] / 100'''

def prepare_prompt_analysis(dimension: str, scenario: str, persona: str, query: str, analysis: str):
    assert dimension.lower() in ["resonation", "expression", "reception"]
    if dimension == "resonation":
        criteria = resonation_criteria
    elif dimension == "expression":
        criteria = expression_criteria
    else:
        criteria = reception_criteria
    prompt = base_prompt_analysis.format(
        dimension=dimension,
        criteria=criteria,
        scenario=scenario,
        persona=persona,
        query=query,
        analysis=analysis
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

def prepare_prompt_response(dimension: str, scenario: str, persona: str, query: str, response: str):
    assert dimension.lower() in ["resonation", "expression", "reception"]
    if dimension == "resonation":
        criteria = resonation_criteria
    elif dimension == "expression":
        criteria = expression_criteria
    else:
        criteria = reception_criteria
    prompt = base_prompt_response.format(
        dimension=dimension,
        criteria=criteria,
        scenario=scenario,
        persona=persona,
        query=query,
        response=response
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

def prepare_prompt_standby(scenario: str, persona: str, query: str, response: str):
    prompt = standby_user_prompt.format(
        scenario=scenario,
        persona=persona,
        query=query,
        response=response
    )
    return [{"role": "system", "content": standby_system_prompt}, {"role": "user", "content": prompt}]

def batch_generate_hf(pipe, prompts: list, batch_size: int = 64, temperature: float = 0.7):
    outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=temperature, batch_size=batch_size)
    return [o[0]['generated_text'][-1]['content'] for o in outputs]

def batch_generate_oai(prompts: list, model: str = "gpt-4o-mini", temperature: float = 0.3, batch_size: int = 64, api_key: str = None, base_url: str = None):
    from api_call import api_call
    # print(prompts[0][-1]['content'])
    outputs = api_call(
        model_name=model, 
        user_prompt_list=prompts, 
        api_list=[api_key], 
        base_url_list=[base_url], 
        api_call_limit=batch_size, 
        temperature=temperature,
        max_completion_tokens=1024,
    )
    # print(outputs)
    return outputs

def batch_generate_vllm(model, prompts: list, batch_size: int = 64, temperature: float = 0.7):
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)
    outputs = model.chat(prompts, sampling_params, use_tqdm=False)
    return [out.outputs[0].text for out in outputs]

