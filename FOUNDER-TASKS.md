# Founder tasks

Things only the repo owner can do: accounts, keys and anything published under his name.
Each one says what it unblocks.

| # | Task | Unblocks | How |
|---|---|---|---|
| 1 | Gemini API key on the free tier | The API baseline row | Create a new Google AI Studio project with no billing account linked (a linked billing account turns free calls into paid ones or into "prepayment credits depleted" errors). Create a key in that project, put it in `.env` as `GEMINI_API_KEY`. Use this key for this project only. |
| 2 | Read the free-tier limits once | The pacing setting and an honest methods note | Open aistudio.google.com/rate-limit for that project and note the RPM, TPM and RPD shown for `gemini-3.5-flash-lite`. Google does not publish them. Until then the client spaces calls 8 seconds apart. |
| 3 | Accept the Jais 2 gate | The Jais 2 row | Sign in to Hugging Face, open huggingface.co/inception42/Jais-2-8B-Chat-GGUF and accept (approval is instant; the gate shares your username and email with Inception). Create a read token and put it in `.env` as `HF_TOKEN`. Then `make models MODEL=jais-2-8b`. |
| 4 | Decide when the repo goes public | Recruiter visibility, the dashboard link | The plan says Friday 27 November. Before flipping it, reread `SCOPE.md`: it publishes the career-goal section and the draft CV bullet as written. |
| 5 | Deploy the dashboard | The GITEX phone link | Vercel, on an account you own. Built after the numbers are final. |
| 6 | Record the video | Output 4 | Saturday 28 November. The first 30 seconds: the problem and the poisoned invoice being routed to review (`make demo-poison` prints the injection, the extraction and the reason). |
| 7 | Post the write-up | Output 6 | One chart, three numbers, the repo link, one question to UAE AI leads. |
| 8 | Optional: the vLLM run on a free Kaggle T4 | A vLLM row, and "self-hosted with vLLM" in the CV line | Needs your Kaggle account and a driver of 580 or later on the notebook. Instructions in `docs/adr/001-models-and-serving.md`. |
