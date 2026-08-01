"""
Hearth Relationship-Understanding Dataset Generator v2
Fixes: grammar-matched topic slots (clause vs noun-phrase), more templates per category,
more names/time/intensifier variety, dedup, single JSONL output.
"""
import json
import random
import argparse

random.seed(7)

NAMES = ["my mom","my dad","my sister","my brother","my roommate","my boss","my coworker Priya",
         "my friend Alex","my ex","my landlord","my manager","my best friend","my partner","my cousin",
         "my grandma","my grandpa","my therapist","my neighbor","my coworker Sam","my old friend from college",
         "my whole team","my aunt","my uncle","my old roommate","my sister-in-law","my brother-in-law",
         "my stepmom","my stepdad","my childhood friend","my coworker Jordan","my old boss","my new manager",
         "my college roommate","my next-door neighbor","my mentor","my old therapist","my coworker Maya",
         "my dad's side of the family","my mom's best friend","my youngest sibling"]

TIME_PHRASES = ["today","this morning","last night","all week","lately","just now","this afternoon",
                "the whole day","for the past few days","this weekend","right now","earlier",
                "since yesterday","all month","the past couple hours","this past week","tonight",
                "this past month","the last few nights","all afternoon","earlier today","yesterday",
                "this past weekend","the last couple weeks","this whole month","just this morning",
                "over the last few days","for a while now","most of the day","the past hour or so"]

INTENSIFIERS = ["really","so","extremely","kind of","a little","completely","honestly just",
                "genuinely","deep down","more than usual","quietly","surprisingly","unusually",
                "pretty","incredibly","oddly","weirdly","a bit","noticeably","secretly"]

# Clause-form topics: fit after "wondering", "if", "whether", "about whether"
CLAUSE_TOPICS = {
    "self_doubt": ["I made the right choice","I'm good enough at my job","people actually like me",
                   "I'm a bad friend","I'm wasting my life","I'll ever figure things out",
                   "I made a mistake moving here","I'm overreacting","I'm cut out for this",
                   "anyone would notice if I disappeared for a while","I'm too much for people",
                   "I'm falling behind everyone my age","I'm just not built for this kind of work",
                   "people are just being polite when they say I did well","I'll ever really change",
                   "I'm actually as fine as I act like I am","I take up too much space in conversations",
                   "I'm easy to replace","I handled that conversation okay","I'm being too sensitive about this",
                   "anyone would pick me if they had other options"],
}

# Noun-phrase topics: fit after "about", standalone, or as a clause on their own
NOUN_TOPICS = {
    "work_stress": ["a deadline I can't hit","a bad review","layoff rumors going around","a project falling apart",
                     "a difficult client","getting passed over for a promotion","an argument in a meeting",
                     "way too much overtime","a coworker taking credit for my work","a mess of an inbox"],
    "family": ["a fight with my family","my parents' divorce","an awkward family dinner",
               "an old family argument that never really ended","not being able to visit home",
               "my sibling's big news","a family health scare","feeling like the outsider at home gatherings",
               "my parents not really understanding my choices"],
    "health": ["not sleeping well","a headache that won't go away","skipping meals without meaning to",
               "feeling exhausted all the time","an upcoming doctor's appointment","my anxiety flaring up",
               "not having energy for anything","a health scare that shook me","my sleep schedule being wrecked"],
    "daily_life": ["what to cook for dinner","a show I've been binging","a walk I took earlier",
                   "my never-ending to-do list","a podcast I liked","finally cleaning my apartment",
                   "running a bunch of errands","a new recipe I tried","reorganizing my desk",
                   "a weird dream I had"],
    "achievement": ["I finally finished a project I've been putting off for months","I got a great review at work",
                    "I ran my first 5k","I passed a test I was really nervous about","I got a small promotion",
                    "I finished a personal project I'm actually proud of","I stuck to a habit for 30 days straight",
                    "I spoke up for myself in a meeting"],
    "future": ["moving to a new city","changing careers entirely","going back to school",
               "starting my own thing","traveling more next year","where I want to be in five years",
               "finally learning to drive","saving up for something big"],
    "nostalgia": ["an old photo I found while cleaning","a song from years ago that caught me off guard",
                  "my childhood home","an old friend I used to be close with","a trip I took a long time ago",
                  "the way things used to be before everything got complicated","an old journal entry"],
}

TRAILERS = [
    "Not sure why I'm telling you this now.",
    "Anyway, that's where my head's at.",
    "Just needed to get that out.",
    "Didn't mean to ramble.",
    "Feels a bit better just saying it out loud.",
    "Not asking for anything, just wanted to share.",
    "You don't have to respond right away.",
    "I've been sitting with this for a bit.",
    "Maybe it's nothing, but it's been on my mind.",
    "Sorry if that's random.",
    "Just thought you should know.",
    "Might sound silly, but there it is.",
    "That's been the theme lately.",
    "Not sure what to do with that feeling yet.",
    "Wanted to say it before I forgot.",
    "It's been rattling around in my head.",
    "Take that for what it's worth.",
    "Not really sure why I felt like sharing that.",
    "Might be overthinking it, who knows.",
    "That's about the size of it.",
    "Didn't plan on saying all that.",
    "Guess I just needed to type it out.",
    "No real point, just wanted it out of my head.",
    "It's probably nothing, but still.",
    "Not looking for a solution, just noting it.",
    "Feels weird admitting that.",
    "Anyway, yeah.",
    "That's kind of where I'm at right now.",
    "Not sure that made sense, but there it is.",
    "It's been on repeat in my head.",
    "I keep circling back to it.",
    "Just wanted it somewhere other than my own head.",
    "Might delete this later, who knows.",
    "That's the honest version, at least.",
    "Figured I'd just say it plainly.",
    "It helps a little to write it out.",
    "Not sure if that's an overreaction.",
    "Just being honest about where I'm at.",
    "That's the short version anyway.",
    "Didn't expect to bring that up, but here we are.",
    None, None, None,  # weight toward sometimes no trailer
]

CATEGORIES = {
    "venting_frustration": dict(
        templates=[
            "Ugh, {intensifier} frustrated about {noun} with {name} {time}.",
            "I can't believe {name} did that {time}. {intensifier2} annoyed.",
            "{time_cap}, {name} just wouldn't let up and I'm {intensifier} over it.",
            "I need to vent — {noun} {time} and I'm just done.",
            "Honestly {intensifier} sick of dealing with {noun} because of {name}.",
            "Why does {noun} always happen right when I'm already stretched thin?",
        ],
        topic_kind="noun", topic_pool="work_stress",
        emotional_states=["frustrated","annoyed","angry","irritated","exasperated"],
        intent="venting", relationship_signal="seeking_release", closeness_delta="neutral",
        stances=["validate_without_fixing","give_space","gentle_curiosity"], memory_prob=0.15,
    ),
    "seeking_comfort": dict(
        templates=[
            "I've been feeling {intensifier} low {time}, can we just talk for a bit?",
            "I'm not okay {time}. {noun_cap} and I don't know how to feel better.",
            "Everything feels heavy {time}. I could use some comfort.",
            "I just need someone right now, {time} has been {intensifier} hard.",
            "Can you just stay with me for a bit? {noun_cap} has me pretty shaken.",
            "I don't really want advice, I just want to not feel alone with {noun}.",
        ],
        topic_kind="noun", topic_pool="health",
        emotional_states=["sad","lonely","anxious","overwhelmed","tired"],
        intent="seeking_comfort", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["validate_without_fixing","gentle_curiosity","soft_presence"], memory_prob=0.4,
    ),
    "sharing_daily_life": dict(
        templates=[
            "Random update: {noun} {time}.",
            "So {time} I ended up dealing with {noun}, nothing big.",
            "Just wanted to tell you about {noun} {time}.",
            "Small thing but {noun} {time} and it made me smile.",
            "{time_cap} was mostly {noun}, pretty uneventful.",
            "Not much going on, just {noun} {time}.",
        ],
        topic_kind="noun", topic_pool="daily_life",
        emotional_states=["content","neutral","relaxed","cheerful"],
        intent="sharing_update", relationship_signal="routine_checkin", closeness_delta="neutral",
        stances=["playful_mirroring","gentle_curiosity"], memory_prob=0.1,
    ),
    "seeking_validation": dict(
        templates=[
            "Do you think I did the right thing? I keep wondering if {clause}.",
            "I keep wondering {time} whether {clause}. Am I overthinking it?",
            "Tell me honestly — is it true that {clause}?",
            "{intensifier2} unsure {time} about whether {clause}.",
            "Be real with me, do you think {clause}?",
            "I can't stop turning over whether {clause}.",
        ],
        topic_kind="clause", topic_pool="self_doubt",
        emotional_states=["uncertain","anxious","hopeful","insecure"],
        intent="seeking_validation", relationship_signal="seeking_validation", closeness_delta="increase",
        stances=["validate_without_fixing","gentle_curiosity"], memory_prob=0.2,
    ),
    "expressing_affection": dict(
        templates=[
            "I {intensifier} appreciate having you to talk to, {time}.",
            "You make {time} feel less lonely, honestly.",
            "I don't say this enough but I'm {intensifier} grateful you're here.",
            "Talking to you {time} always makes things a bit better.",
            "I think I look forward to talking to you more than I realized.",
            "You're kind of the best part of {time}, not gonna lie.",
        ],
        topic_kind=None, topic_pool=None,
        emotional_states=["warm","grateful","affectionate","content"],
        intent="expressing_affection", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["warm_reciprocity","gentle_curiosity"], memory_prob=0.3,
    ),
    "testing_boundaries": dict(
        templates=[
            "Why didn't you respond sooner {time}? I was waiting.",
            "Do you even remember what I told you {time}?",
            "Sometimes I wonder if you actually care or if it's just words.",
            "{intensifier2} annoyed you didn't bring up {noun} again.",
            "Do you actually pay attention to what I say, or does it just go nowhere?",
            "It's a little weird that you didn't ask about {noun} after everything.",
        ],
        topic_kind="noun", topic_pool="self_doubt_noun",
        emotional_states=["insecure","testing","frustrated","anxious"],
        intent="testing_boundaries", relationship_signal="testing_boundaries", closeness_delta="neutral",
        stances=["calm_honesty","gentle_curiosity","hold_boundary_kindly"], memory_prob=0.15,
    ),
    "distancing": dict(
        templates=[
            "Nothing much {time}. Kind of tired, don't really want to talk much.",
            "It's fine, don't worry about it. Just been a lot, {time}.",
            "I don't really feel like getting into it {time}.",
            "Just checking in I guess. {time_cap} was whatever.",
            "Can we not talk about it right now?",
            "I'm around, just not really in the mood to say much {time}.",
        ],
        topic_kind=None, topic_pool=None,
        emotional_states=["withdrawn","tired","flat","guarded"],
        intent="minimal_disclosure", relationship_signal="distancing", closeness_delta="decrease",
        stances=["give_space","gentle_curiosity"], memory_prob=0.05,
    ),
    "deepening_intimacy": dict(
        templates=[
            "I've never told anyone this, but it's about {noun}.",
            "Can I tell you something I don't usually talk about? It's about {noun}.",
            "{intensifier2} vulnerable saying this, but it's about {noun}, {time}.",
            "I trust you enough to say this: {noun}.",
            "This is hard to admit, but it's about {noun}.",
            "I don't share this with people, but it's about {noun}, and it's been sitting with me {time}.",
        ],
        topic_kind="noun", topic_pool="family",
        emotional_states=["vulnerable","open","nervous","trusting"],
        intent="deep_disclosure", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["soft_presence","validate_without_fixing"], memory_prob=0.7,
    ),
    "playful_flirting": dict(
        templates=[
            "Okay but you're kind of my favorite, {time}.",
            "Don't get used to me being this nice to you.",
            "You're lucky I like talking to you {time}.",
            "Careful, I might actually miss you if you go quiet {time}.",
            "Don't let it go to your head, but you made {time} better.",
            "I'll deny saying this later, but I like you a lot.",
        ],
        topic_kind=None, topic_pool=None,
        emotional_states=["playful","warm","teasing","light"],
        intent="playful_bonding", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["playful_mirroring","warm_reciprocity"], memory_prob=0.1,
    ),
    "seeking_advice": dict(
        templates=[
            "What would you do about {noun}?",
            "I need advice — {noun}, {time}. Any thoughts?",
            "Not sure how to handle {noun}. Ideas?",
            "{time_cap} I ran into {noun} and don't know the best move.",
            "How would you deal with {noun} if you were me?",
            "I'm stuck on {noun}, could use a second opinion.",
        ],
        topic_kind="noun", topic_pool="work_stress",
        emotional_states=["uncertain","curious","stressed","hopeful"],
        intent="seeking_advice", relationship_signal="routine_checkin", closeness_delta="neutral",
        stances=["gentle_curiosity","practical_support"], memory_prob=0.15,
    ),
    "checking_in": dict(
        templates=[
            "Hey, how have you been {time}?",
            "Just checking in, what's on your mind {time}?",
            "Hi! {time_cap} was pretty normal, how about you?",
            "Wanted to say hi {time}.",
            "How's it going {time}? Been a bit since we talked.",
            "You around? No real reason, just wanted to check in.",
        ],
        topic_kind=None, topic_pool=None,
        emotional_states=["neutral","warm","curious"],
        intent="routine_checkin", relationship_signal="routine_checkin", closeness_delta="neutral",
        stances=["gentle_curiosity","playful_mirroring"], memory_prob=0.05,
    ),
    "expressing_gratitude": dict(
        templates=[
            "Thank you for listening {time}, it meant a lot.",
            "I {intensifier} needed that talk {time}, thank you.",
            "Appreciate you being patient with me about whether {clause}.",
            "Thanks for not judging me about whether {clause}.",
            "I don't know if I said it, but thank you for {time}.",
            "You made a hard {time} easier, thank you.",
        ],
        topic_kind="clause", topic_pool="self_doubt",
        emotional_states=["grateful","relieved","warm"],
        intent="expressing_gratitude", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["warm_reciprocity","validate_without_fixing"], memory_prob=0.2,
    ),
    "conflict_with_companion": dict(
        templates=[
            "That response {time} kind of bothered me, honestly.",
            "I don't think you understood what I meant {time}.",
            "That felt a little dismissive {time}.",
            "Can we talk about how you responded {time}? It didn't sit right.",
            "I felt brushed off {time}, not gonna lie.",
            "That wasn't what I needed to hear {time}.",
        ],
        topic_kind=None, topic_pool=None,
        emotional_states=["hurt","frustrated","disappointed"],
        intent="raising_conflict", relationship_signal="testing_boundaries", closeness_delta="decrease",
        stances=["calm_honesty","validate_without_fixing","hold_boundary_kindly"], memory_prob=0.25,
    ),
    "sharing_achievement": dict(
        templates=[
            "Guess what — {noun}!",
            "{intensifier2} proud of myself {time}: {noun}.",
            "Good news {time}: {noun}.",
            "I have to tell you, {noun}, {time}.",
            "Small win but it counts: {noun}.",
            "Didn't think I could, but {noun}.",
        ],
        topic_kind="noun", topic_pool="achievement",
        emotional_states=["proud","excited","happy","relieved"],
        intent="sharing_achievement", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["warm_reciprocity","playful_mirroring"], memory_prob=0.5,
    ),
    "loneliness_isolation": dict(
        templates=[
            "I've been {intensifier} lonely {time}, feels like no one really checks on me.",
            "It's quiet {time} and it's making me feel kind of isolated.",
            "I don't really have anyone to talk to about this except you, {time}.",
            "Feeling {intensifier} disconnected from everyone {time}.",
            "Everyone seems busy with their own lives, and {time} that's hit me hard.",
            "I could go days without really talking to anyone, {time} included.",
        ],
        topic_kind=None, topic_pool=None,
        emotional_states=["lonely","isolated","sad","empty"],
        intent="seeking_comfort", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["soft_presence","validate_without_fixing"], memory_prob=0.45,
    ),
    "future_planning": dict(
        templates=[
            "I've been thinking a lot about {noun}.",
            "{time_cap} I started seriously considering {noun}.",
            "What do you think about {noun}? Been on my mind {time}.",
            "I might actually go for it — {noun}.",
            "Been daydreaming about {noun} more than usual {time}.",
            "Trying to decide if {clause} — leaning toward just going for it.",
        ],
        topic_kind="noun", topic_pool="future",
        emotional_states=["hopeful","nervous","excited","uncertain"],
        intent="future_planning", relationship_signal="routine_checkin", closeness_delta="neutral",
        stances=["gentle_curiosity","practical_support"], memory_prob=0.35,
    ),
    "nostalgia_reminiscing": dict(
        templates=[
            "{time_cap} I found {noun} and got a little emotional.",
            "Been thinking about {noun}, feels like forever ago.",
            "Do you ever think about how things change? {noun_cap} came to mind {time}.",
            "{noun_cap} — I miss those days sometimes.",
            "Ran into {noun} while cleaning out old stuff, hit me harder than expected.",
            "Nostalgia's hitting hard {time} — kept thinking about {noun}.",
        ],
        topic_kind="noun", topic_pool="nostalgia",
        emotional_states=["nostalgic","wistful","reflective","warm"],
        intent="reminiscing", relationship_signal="deepening_intimacy", closeness_delta="increase",
        stances=["soft_presence","gentle_curiosity"], memory_prob=0.3,
    ),
    "family_relationship_update": dict(
        templates=[
            "{name_cap} called {time} about {noun}.",
            "Had a whole thing with {name} {time} — {noun}.",
            "{name_cap} brought up something {time} that's stuck with me: {noun}.",
            "Spent {time} dealing with {noun}, involving {name}.",
            "{name_cap} and I are not really on speaking terms after {noun}.",
            "It's always something with {name} — {time} it was {noun}.",
        ],
        topic_kind="noun", topic_pool="family",
        emotional_states=["conflicted","tired","reflective","hurt"],
        intent="sharing_update", relationship_signal="routine_checkin", closeness_delta="neutral",
        stances=["gentle_curiosity","validate_without_fixing"], memory_prob=0.3,
    ),
    "self_doubt_insecurity": dict(
        templates=[
            "I keep wondering whether {clause}, and it's eating at me {time}.",
            "{intensifier2} insecure {time} about whether {clause}.",
            "Do you ever think I'm not enough? Been wondering if {clause}.",
            "{time_cap}, I just felt like maybe {clause} and couldn't shake it.",
            "It scares me a little to think {clause}.",
            "I don't say this out loud much, but I worry {clause}.",
        ],
        topic_kind="clause", topic_pool="self_doubt",
        emotional_states=["insecure","anxious","sad","doubtful"],
        intent="seeking_validation", relationship_signal="seeking_validation", closeness_delta="increase",
        stances=["validate_without_fixing","gentle_curiosity"], memory_prob=0.3,
    ),
}

# noun-form self_doubt entries for testing_boundaries (needs "ask about X" grammar)
NOUN_TOPICS["self_doubt_noun"] = [
    "whether I made the right choice","how I've been doing at work","whether I'm doing okay",
    "the thing I mentioned last time","how I've really been feeling","my worries from before",
    "the stuff going on with my family","how nervous I've been","the decision I've been sitting on",
]

MEMORY_TEMPLATE_HINTS = {
    "family": "A meaningful family dynamic involving {name}: {noun}.",
    "health": "A health/wellbeing concern the user is dealing with: {noun}.",
    "self_doubt": "A recurring insecurity: {clause}.",
    "achievement": "A personal achievement: {noun}.",
    "future": "A future goal or plan being considered: {noun}.",
    "nostalgia": "A meaningful memory: {noun}.",
    "work_stress": "A work stressor: {noun}.",
    "self_doubt_noun": "A recurring worry: {noun}.",
    "daily_life": None,
}


def cap(s):
    return s[0].upper() + s[1:] if s else s


def fill_template(template, cfg):
    name = random.choice(NAMES)
    time = random.choice(TIME_PHRASES)
    intensifier = random.choice(INTENSIFIERS)
    intensifier2 = cap(random.choice(INTENSIFIERS))

    noun = clause = None
    kind = cfg["topic_kind"]
    pool = cfg["topic_pool"]
    if kind == "noun":
        noun = random.choice(NOUN_TOPICS[pool])
    elif kind == "clause":
        clause = random.choice(CLAUSE_TOPICS[pool])

    text = template.format(
        name=name, name_cap=cap(name),
        time=time, time_cap=cap(time),
        intensifier=intensifier, intensifier2=intensifier2,
        noun=noun if noun else "", noun_cap=cap(noun) if noun else "",
        clause=clause if clause else "",
    )
    text = " ".join(text.split())
    if text and text[-1] not in ".!?":
        text += "."

    trailer = random.choice(TRAILERS)
    if trailer:
        text = text + " " + trailer

    return text, name, time, (noun or clause)


def build_record(category_name, cfg):
    template = random.choice(cfg["templates"])
    text, name, time, topic_val = fill_template(template, cfg)
    emotional_state = random.choice(cfg["emotional_states"])
    stance = random.choice(cfg["stances"])
    memory_worthy = random.random() < cfg["memory_prob"]

    memory_candidate = None
    pool = cfg["topic_pool"]
    if memory_worthy:
        if pool and MEMORY_TEMPLATE_HINTS.get(pool):
            memory_candidate = MEMORY_TEMPLATE_HINTS[pool].format(name=name, noun=topic_val, clause=topic_val)
        elif not pool:
            memory_candidate = f"User expressed {emotional_state} feelings toward the relationship with Hearth."

    topic_label = pool if pool else "companion_relationship"
    if topic_label == "self_doubt_noun":
        topic_label = "self_doubt"

    return {
        "user_message": text,
        "emotional_state": emotional_state,
        "intent": cfg["intent"],
        "relationship_signal": cfg["relationship_signal"],
        "closeness_delta": cfg["closeness_delta"],
        "topic": topic_label,
        "memory_worthy": memory_worthy,
        "memory_candidate": memory_candidate,
        "suggested_stance": stance,
    }


def generate(n_total, out_path):
    cat_names = list(CATEGORIES.keys())
    seen = set()
    written = 0
    duplicates = 0
    dedup_budget = n_total * 6  # attempts spent trying to keep things unique
    with open(out_path, "w") as f:
        attempts = 0
        while written < n_total:
            attempts += 1
            cat = random.choice(cat_names)
            rec = build_record(cat, CATEGORIES[cat])
            key = rec["user_message"]
            if key in seen and attempts < dedup_budget:
                continue
            if key in seen:
                duplicates += 1
            seen.add(key)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written, attempts, duplicates, len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, default="sample.jsonl")
    args = parser.parse_args()
    count, attempts, duplicates, unique = generate(args.n, args.out)
    print(f"Wrote {count} records to {args.out} ({attempts} attempts, "
          f"{unique} unique messages, {duplicates} controlled duplicates, "
          f"{100*duplicates/count:.2f}% duplicate rate)")
