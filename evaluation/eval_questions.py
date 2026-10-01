"""
eval_questions.py

Expanded evaluation set: 15 AI/ML questions with hand-written reference
(ground-truth) answers -- written by us, not LLM-generated, specifically to
avoid the circularity of using an LLM to grade against an LLM-written
"truth." These are standard, well-established facts about AI/ML concepts,
chosen to span topics we know are represented in the crawled Wikipedia
"Artificial Intelligence" category graph (machine learning, generative AI,
NLP, neural networks, AI safety, computer vision, reinforcement learning,
expert systems, knowledge representation, evolutionary computation).
"""

QA_PAIRS = [
    {
        "question": "What is generative AI and how does it relate to large language models?",
        "ground_truth": (
            "Generative AI refers to systems that create new content (text, images, "
            "audio, code) rather than just classifying or predicting labels. Large "
            "language models are a major type of generative AI: they are trained on "
            "large text corpora to predict and generate coherent text, and are the "
            "technology behind tools like ChatGPT."
        ),
    },
    {
        "question": "What are the risks and safety concerns around artificial intelligence?",
        "ground_truth": (
            "Key AI safety concerns include misalignment (systems pursuing goals "
            "different from human intentions), misuse (deliberate harmful "
            "applications), unpredictable behavior in high-stakes settings, "
            "job displacement, bias and fairness issues, and longer-term "
            "existential risk from highly capable future systems."
        ),
    },
    {
        "question": "How is natural language processing used in chatbots?",
        "ground_truth": (
            "Chatbots use NLP to understand user input (natural language "
            "understanding), determine intent (classification), and produce "
            "coherent responses (natural language generation). Modern chatbots "
            "like ChatGPT use large language models built on the transformer "
            "architecture to do all of this in one unified system."
        ),
    },
    {
        "question": "What is the history of artificial intelligence research?",
        "ground_truth": (
            "AI research began formally at the 1956 Dartmouth workshop. It went "
            "through cycles of optimism and 'AI winters' (funding cutbacks) in the "
            "1970s and late 1980s, followed by a resurgence driven by machine "
            "learning in the 1990s-2000s, and a deep-learning-driven boom from "
            "the early 2010s onward, accelerating further with large language "
            "models after 2020."
        ),
    },
    {
        "question": "What are neural networks and how do they learn?",
        "ground_truth": (
            "Artificial neural networks are computational models loosely inspired "
            "by biological neurons, consisting of layers of interconnected nodes "
            "with adjustable weights. They learn via backpropagation: computing "
            "the error between predicted and actual outputs, then adjusting "
            "weights via gradient descent to reduce that error over many training "
            "examples."
        ),
    },
    {
        "question": "What is reinforcement learning and how does it differ from supervised learning?",
        "ground_truth": (
            "Reinforcement learning trains an agent to make sequential decisions "
            "by rewarding or penalizing actions based on outcomes, learning a "
            "policy that maximizes cumulative reward through trial and error. "
            "This differs from supervised learning, which learns from a fixed "
            "dataset of labeled input-output examples rather than interactive "
            "feedback."
        ),
    },
    {
        "question": "What is computer vision and what tasks does it solve?",
        "ground_truth": (
            "Computer vision is the field of AI concerned with enabling computers "
            "to interpret visual information from images and video. Common tasks "
            "include image classification, object detection, facial recognition, "
            "image segmentation, and optical character recognition."
        ),
    },
    {
        "question": "What is an expert system in artificial intelligence?",
        "ground_truth": (
            "An expert system is an AI program that emulates the decision-making "
            "of a human domain expert, typically using a knowledge base of facts "
            "and rules combined with an inference engine to reason about them and "
            "give recommendations, common in early AI applications like medical "
            "diagnosis."
        ),
    },
    {
        "question": "What is knowledge representation in AI?",
        "ground_truth": (
            "Knowledge representation is the area of AI concerned with encoding "
            "information about the world in a form a computer can use to solve "
            "complex tasks, using structures such as semantic networks, "
            "ontologies, frames, and logic-based rules."
        ),
    },
    {
        "question": "What is evolutionary computation?",
        "ground_truth": (
            "Evolutionary computation is a family of AI algorithms inspired by "
            "biological evolution, including genetic algorithms, that iteratively "
            "generate, evaluate, and recombine candidate solutions using "
            "selection, mutation, and crossover to improve solution quality over "
            "generations."
        ),
    },
    {
        "question": "What is the difference between narrow AI and general AI?",
        "ground_truth": (
            "Narrow AI (or weak AI) is designed to perform a specific task, such "
            "as image recognition or language translation, and cannot generalize "
            "beyond it. General AI (or strong AI/AGI) refers to a hypothetical "
            "system with human-level cognitive ability across essentially any "
            "intellectual task, which does not currently exist."
        ),
    },
    {
        "question": "What role do transformers play in modern AI models?",
        "ground_truth": (
            "The transformer is a neural network architecture introduced in 2017 "
            "that uses self-attention to weigh the relevance of different parts "
            "of the input to each other, allowing it to process sequences in "
            "parallel rather than step-by-step. It is the foundational "
            "architecture behind most modern large language models."
        ),
    },
    {
        "question": "What is machine learning and how does it differ from traditional programming?",
        "ground_truth": (
            "Machine learning is a subfield of AI in which systems learn patterns "
            "from data rather than being explicitly programmed with rules. "
            "Traditional programming requires a human to write explicit logic for "
            "every case, while machine learning infers a model from example data "
            "that can generalize to new, unseen cases."
        ),
    },
    {
        "question": "What ethical concerns are commonly discussed regarding AI?",
        "ground_truth": (
            "Common AI ethics concerns include algorithmic bias and discrimination, "
            "privacy violations from data collection, lack of transparency in "
            "decision-making ('black box' models), accountability for AI-caused "
            "harm, and the concentration of power in organizations that control "
            "powerful AI systems."
        ),
    },
    {
        "question": "What is a deepfake and how is it created using AI?",
        "ground_truth": (
            "A deepfake is synthetic media in which a person's likeness or voice "
            "is replaced or generated using deep learning, typically involving "
            "generative adversarial networks (GANs) or other generative models "
            "trained on real footage or audio of the target person to produce "
            "realistic but fabricated content."
        ),
    },
]
