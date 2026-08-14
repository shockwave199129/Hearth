# Hearth Governance

This document describes how decisions get made in Hearth today, and how
that's meant to change as the project grows. It exists so nobody has to
guess, and so the project's direction is never dependent on private
conversations or unwritten assumptions.

## Where we are today: Founder-led, with a written path to shared control

Hearth is currently maintained by its original author, Subhrangshu Mondal
(**@shockwave199129**), who makes final calls on architecture, releases,
and scope. This is deliberate at this stage — the project is small, the
core design (privacy-first, fully local, no telemetry) needs a consistent
hand while it's still being defined, and most decisions are still
reversible.

This is **not** meant to be permanent, and it is not a claim of ownership
over the project. See "Licensing and ownership" below for why that
distinction is real and enforceable, not just a promise.

## Where we're going: Core team governance

As contributors put in sustained, high-quality work, they'll be invited to
become **co-maintainers** with commit access and a vote on project
direction. There's no fixed contribution quota — the bar is judgment,
consistency, and a track record other contributors trust. Typical signals:

- Multiple substantive, merged PRs across more than one area of the codebase
- Thoughtful code review of other people's PRs, not just your own
- Helping newer contributors get unblocked
- Understanding *why* the project makes the choices it does (privacy-first,
  local-only, no dark patterns), not just its code

Once there are 3+ active co-maintainers, we move to **lazy consensus with a
maintainer vote for contested decisions** (the model most mid-size open
source projects converge on, borrowed from Apache-style governance): a
change ships if no maintainer objects within a review window; if someone
objects, maintainers vote and majority decides.

## The succession commitment

The long-term intent for Hearth is the same trajectory as projects like
Linux or Git: founder-led at the start, community-governed later, never
sold to a single company for exclusive control. Concretely:

1. **No exclusive sale of the project or its trademark is planned or
   intended.** Anyone can already fork Hearth today under its license —
   that's not a courtesy, it's structural (see below).
2. When the core team is established and self-sustaining, maintainership
   and the "Hearth" name/trademark are intended to transfer to a neutral
   nonprofit fiscal host for independent open source projects (e.g.
   Software Freedom Conservancy), the way Git, Inkscape, and QEMU have
   done. This removes any single person's ability to unilaterally change
   the license or governance later.
3. Until that transfer happens, this file is the record of intent, and any
   deviation from it should be treated as a legitimate reason for the
   community to fork.

## Licensing and ownership (what is and isn't enforceable)

Hearth is licensed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)),
and takes contributions under a **Developer Certificate of Origin (DCO)**
rather than a Contributor License Agreement (CLA). It's worth being precise
about what that combination does and doesn't protect, because the two are
often conflated.

**What it does guarantee — no single entity accumulates ownership.** Under a
CLA, contributors typically assign or broadly license their copyright to the
project owner, who can then relicense the whole project — including to a
single company, exclusively, later. That's how several well-known "open
source" projects were eventually sold or closed. Under DCO, **each
contributor retains copyright over their own contribution**. Nobody,
including the original author, holds enough rights to claim sole ownership
of the codebase or to sell exclusive rights to it. This is the same
mechanism Linux relies on.

**What it also guarantees — the published code stays open, permanently.**
The Apache 2.0 grant on every released version is irrevocable. No future
maintainer, company, or governance change can retract it. Anyone can fork
what has already shipped, forever. That's the structural fork right
referenced above.

**What it does not guarantee — that every derivative stays open.** Apache
2.0 is a permissive license, not a copyleft one. Anyone — a company, or any
maintainer here — may fork Hearth and ship a closed-source derivative
without publishing their changes. AGPLv3 would prevent that; Apache 2.0
does not. This is a deliberate trade in favour of adoption and downstream
reuse, and it's stated plainly here rather than implied away.

So: if you contribute to Hearth, your code stays yours, and this project
can never be taken private out from under you. A *fork* of it can be closed,
and that is a real limitation of the license we chose.

## Commercial use

Hearth's open core will always remain free and open under Apache 2.0. Paid
offerings (e.g. a packaged installer, cloud sync, premium models/support)
are separate, optional add-ons layered on top — not a paywall on safety
features, core functionality, or your right to run, audit, and modify the
software yourself. See `PRICING.md` (once published) for specifics.

## Changing this document

Right now, changes to GOVERNANCE.md are made by the founding maintainer.
Once a core team exists, changes require maintainer consensus. Either way,
changes are made via a public, reviewable pull request — never silently.
