export const meta = {
  name: 'corpus-judgment-tagging',
  description: 'Judgment-tag pending corpus MIDI with the VGM dramatic-type taxonomy (one agent per batch)',
  phases: [{ title: 'Tag', detail: 'one agent per batch of pending files; writes a tags JSON per batch' }],
}

// batch files at /tmp/cat_batches/batch_<i>.jsonl (zero-padded), outputs to /tmp/cat_out/batch_<i>.json
const N = 46

const TAXONOMY = `
  battle     - active combat, driving
  boss       - high-stakes fight, grand/menacing
  overworld  - field / map / exploration travel
  town       - settlement, peaceful hub, shop
  dungeon    - cave / fortress, dangerous exploration
  character  - a personality leitmotif
  romance    - love, tender, intimate
  sad        - loss, lament, tragic
  victory    - fanfare, triumph, win jingle
  tension    - danger, suspense, unease
  title      - title screen, menu, intro
  ending     - credits, finale, farewell
  ambient    - atmosphere, mystery, eerie
  comedy     - quirky, lighthearted, playful
  heroic     - noble, grand, resolute
  march      - processional, military, steady tread
  dance      - waltz / minuet / jig, lilting
  chorale    - hymn, sacred, solemn block harmony
  fugue      - contrapuntal, intricate, learned
  virtuosic  - showy, technical, fast
  aria       - lyrical, melody-forward, singable
  nocturne   - calm, reflective, night`

const pad = (i) => String(i).padStart(3, '0')
const prompt = (i) => `You are categorizing video-game / classical MIDI pieces for a symphony's training corpus.
Apply ONE unified taxonomy to every piece through a VGM dramatic lens: "what setting / scene would this score?"

THE ONLY VALID TAGS (use these exact strings, nothing else):
${TAXONOMY}

Your batch is the JSONL file: /tmp/cat_batches/batch_${pad(i)}.jsonl
Each line is one piece: {"path","category"(=corpus bucket / game folder),"filename","evidence"}.
evidence = {key, tempo(bpm), density(notes/beat), mean_interval(semitones), range(semitones), notes(count)}.
"error" in evidence means the file failed to parse.

For EACH piece, judge from filename (often the strongest cue — e.g. "boss", "town2", "ending", "battle"),
the game/category context, and the musical evidence (mode/tempo/density/angularity/range). Decide:
  - primary: 1-2 best-fit tags (the dominant function/mood)
  - supporting: 0-3 other tags that also apply (flavours)
  - verdict: "keep" normally; "reject" ONLY for non-training material — failed-parse/empty evidence,
    percussion/SFX-only, jingles under ~8 notes, or clearly non-musical. When rejecting, primary=[].
  - reason: a short phrase (<= 12 words) justifying the call.

Read the batch file, judge all of its lines, and WRITE your answer as a single JSON object to
/tmp/cat_out/batch_${pad(i)}.json mapping each piece's "path" to {"primary":[...],"supporting":[...],"verdict":"keep|reject","reason":"..."}.
Use the Write tool. Cover EVERY line in the batch — do not skip any. Only use tags from the list above.
Return one line: "batch ${i}: <kept> kept, <rejected> rejected".`

log(`tagging ${N} batches of pending corpus files`)
const results = await parallel(
  Array.from({ length: N }, (_, i) => () =>
    agent(prompt(i), { label: `tag:batch_${i}`, phase: 'Tag' })
  )
)
const done = results.filter(Boolean).length
log(`agents finished: ${done}/${N}`)
return { batches: N, agents_completed: done, summaries: results }
