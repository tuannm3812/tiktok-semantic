# Competition Insight Brief

## Executive Answer

Climate-action TikToks appear most effective when they translate urgency into a concrete action frame. In the current sample, the strongest creative combinations are not generic climate-crisis warnings alone; they pair an emotionally clear hook with a practical path such as personal eco-lifestyle changes, public transit, political accountability, or an easily understood climate analogy.

Recommended marketing strategy:

1. Lead with a vivid hook that makes climate feel immediate.
2. Convert attention into a clear action frame within the post.
3. Use comments as a research channel for audience barriers, questions, and objections.
4. Prioritize creators whose historical reach or network position overperforms their follower count.

## Evidence Snapshot

The processed local sample contains 10 posts from 2023-01-02 to 2025-05-24, with 5.43M total views, 743K likes, 25.7K shares, and 710 captured comments.

Because this is a small sample, the findings should be presented as directional evidence and a repeatable analytical framework rather than final population estimates. The scripts are designed to rerun on the full 1,597-post dataset once it is available locally.

## 1. Format: Video Carries Reach, Carousel Formats Can Carry Action

Video posts have the strongest median reach in the sample: 889.6K median views across 5 video posts, compared with 391.9K for image-plus-audio posts and 1.0K for image-plus-music posts.

However, the top engagement-efficiency post is an image-plus-music eco-lifestyle carousel: 212.3K views and 1,270 engagements per 1K views. That suggests carousel-style posts can work well when the message is step-by-step, saveable, and personally actionable.

Marketing implication: use video for mass awareness and carousel formats for action checklists, habit formation, and save/share behavior.

## 2. Message Framing: Urgency Works Best When It Has Direction

The strongest high-reach post uses an urgent, awareness-led environmental message connected to Earth's beauty: 2.60M views and 11.7K shares. The strongest efficiency post uses a hopeful, empowering individual-action frame: 1,270 engagements per 1K views.

The pattern is useful: urgency attracts attention, but action framing gives people something to do with that attention. Pure crisis framing without a strong creative bridge performs weakly in the sample, especially in low-reach 2025 posts.

Marketing implication: avoid "climate crisis" as a standalone message. Pair it with one of four action routes: individual habit, collective behavior, policy pressure, or practical learning.

## 3. Audience Response: Comments Reveal Conversion Barriers

Captured comments are not just applause. The comment intent table surfaces practical barriers and conversion questions:

- 116 comments are classified as barrier or constraint signals.
- 100 comments are action-oriented.
- 42 comments are explicit information questions.
- Only 3 comments are classified as skeptic or counter-argument signals.

Examples include viewers asking whether gardens help, whether eating less meat is enough, or saying action is hard because they are young, at school, or dependent on parents.

Marketing implication: climate creators should treat comment sections as message testing. The next post should answer the most common barrier: "What can someone like me realistically do?"

## 4. Creator Strategy: Smaller Creators Can Overperform

Creator history suggests follower count alone is not a sufficient planning metric. One creator with 5.98K followers has a historical average of about 2.60M views for the captured post history, while a mega-scale account with 2.40M followers averages about 2.25M views.

Marketing implication: prioritize creators by historical views per follower, not just follower count. A climate campaign should identify high-efficiency creators who consistently overperform their audience size.

## 5. Creator Networks: Bridge Position Matters

The creator bridge table uses captured follower and following relationships to estimate network degree, betweenness, component size, and a blended bridge score. In the sample, high bridge scores are not identical to high follower counts. This creates a more useful campaign activation lens: who can carry climate messages across adjacent communities?

Marketing implication: use bridge creators to move climate content from climate-native audiences into lifestyle, politics, transport, and entertainment audiences.

## 6. Hashtags: Broad Climate Tags Need Context Tags

`#climatechange` appears in 9 of 10 posts and drives most aggregate volume, but it is too broad to explain performance by itself. The stronger analytical signal comes from pairing broad tags with contextual tags:

- Entertainment/attention tags: `#fyp`, `#avatar2`, `#interview`
- Action-context tags: `#ecofriendly`, `#sustainableliving`, `#publictransportation`
- Issue-specific tags: `#globalwarming`, `#climatecrisis`, `#carbonemissions`

Marketing implication: use broad climate tags for discoverability, but attach a context tag that tells TikTok and viewers what kind of climate story the post is: lifestyle, transport, policy, extreme weather, or pop-culture analogy.

## Deep-Dive Extension Status

1. Full 1,597-post run: not completed in this repo yet because the available local/Drive data only contains the 10-post `Data_Sample`. The pipeline is ready for the full dataset once the full `dtaPosts.pkl` and `Videos/` folder are placed under `data/raw`.
2. Semantic clustering: completed as a dependency-light TF-IDF/KMeans version in `scripts/deep_dive_nlp.py`. On the current sample it finds two narrative clusters:
   - Video-led Earth/politics/interview cluster: 5 posts, 4.43M total views, 889.6K median views.
   - Image/audio climate explainer cluster: 5 posts, 998K total views, 1.0K median views.
3. Comment sentiment and emotion: completed as a transparent lexicon/rule-based version in `scripts/deep_dive_nlp.py`. Current comment outputs show that general neutral reaction dominates, while confusion and practical questions are the largest actionable second layer.
4. Creator network features: completed for the available sample through `creator_bridge_metrics.csv`, which uses captured follower/following edges to estimate degree, betweenness, component size, and a blended bridge score.
5. Creative playbook: partly complete through `messaging_recommendations.csv` and this insight brief. The next version should turn top patterns into slide-ready campaign templates.

## Additional NLP Findings

The semantic clustering reinforces the format finding. The higher-reach cluster is video-led and built around dramatic Earth, politics, and interview-style narratives. The lower-reach cluster contains image/audio explainers, including car-free transit and extreme-weather education. This does not mean explainers are weak; it suggests they may need stronger hooks or creator distribution to travel as far as video-led climate narratives.

Comment sentiment and emotion show a conversion opportunity. The largest category is neutral/general reaction, but `confusion` appears frequently and information questions receive high average likes. That means educational follow-up content can be designed directly from comments: answer "what should I do?", "does this action help?", and "how can I participate if my choices are constrained?"

## Additional Sample Insights

The sample supports four campaign hypotheses worth testing on the full dataset:

1. **Action beats awareness when the goal is efficient engagement.** The highest engagement-efficiency post is a practical eco-lifestyle carousel, not the highest-reach video. This points to a two-stage funnel: use high-emotion video to earn reach, then use practical formats to convert attention into action.
2. **Questions are high-value comments.** Explicit information questions are fewer than general reactions, but they attract high average comment likes. That suggests question-answer follow-up videos could perform well because they respond to visible audience demand.
3. **Bridge creators are not always the biggest creators.** The bridge score surfaces creators with strong network position and/or captured follower/following reach. This is a better activation shortlist than follower count alone because climate communication often needs to cross from climate-native audiences into lifestyle, politics, transport, and entertainment spaces.
4. **Broad climate hashtags need narrative context.** `#climatechange` supplies discoverability, but the stronger strategic signal comes from paired context tags such as public transport, eco-lifestyle, interviews, policy, extreme weather, or pop-culture references.

Recommended presentation framing: **Reach comes from hooks, persuasion comes from frames, and conversion opportunities come from comments.**

## Next Analysis Moves

1. Add the full 1,597-post dataset to `data/raw`, then rerun `python scripts/build_dataset.py --config configs/sample.yaml` and `python scripts/deep_dive_nlp.py --config configs/sample.yaml`.
2. Install optional NLP dependencies and rerun clustering with sentence-transformer embeddings when scale justifies it.
3. Replace lexicon sentiment with transformer sentiment/emotion for final presentation robustness.
4. Expand follower/following graph features on the full dataset to identify bridge creators across climate, lifestyle, politics, entertainment, and transport audiences with more stable network centrality estimates.
