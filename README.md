# Lab 3: Choosing an audit target
A good audit target begins with a hypothesis, or at least an interest in a question, about what types of results you're likely to receive from an AI system. This might be a bias audit, like some of the discrimination concerns we've discussed in the class, or focused on the accuracy of the system or on what types of responses it's likely to generate in a specific domain. Whatever your hypothesis, you'll need to demonstrate that you have access to run chosen and controlled queries through an AI system (e.g., via an API) or that you have access to collect a large set of both inputs and outputs previously run through the system that will allow you to answer your question. You should run a pilot study with a small amount of data and/or prompts to determine how your study will work and set any parameters of the study (e.g., prompt engineering).

## What to hand in
1. A description in this README of your audit target and hypothesis, as well as a model card describing (and citing) as much of the information about your audit target as you can find.
2. A `pilot.py` which when run as `python pilot.py` runs your pilot experiment. If you are using a paid API, mention that in your README and we won't run the pilot script (we'll just read it), but you should still make sure it correctly runs.
3. A description in this README of your pilot experiment results. If your results include visualizations (great!) then you should also include an `analysis.py` which when run with `python analysis.py` reads in any data from your pilot experiment and generates any visualizations. Be sure to also commit and push the resulting visualizations to your repository.

# AI use
*Below, describe how you used AI to help you in writing the code for this lab, if at all.*
