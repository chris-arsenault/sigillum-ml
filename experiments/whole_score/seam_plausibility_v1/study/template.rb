production_piece "CASE_TITLE", id: :CASE_ID do
  METER_DECLARATION
  key "CASE_KEY"

  material :theme do
    identity pitch: "Movement IV source excerpt",
             rhythm: "four-measure source phrase"
  end

  roster do
    part :flute, "Flute", music21: "Flute", family: :woodwind
    part :clarinet, "Clarinet", music21: "Clarinet", family: :woodwind
    part :harp, "Harp", music21: "Harp", family: :plucked
  end

  section :whole, "Statement and continuation", bars: 1..8, type: :binary do
    span :statement, bars: 1..4, texture: :melody_over_bass do
      plan do
        requires :material, :theme, relation: :statement
        requires :role, :foreground, coverage: :all_bars
        requires :role, :counterline
        requires :role, :color_anchor
      end
      # ANCHOR_CONTENT
    end

    span :continuation, bars: 5..8, texture: :varied_return do
      plan do
        requires :material, :theme, relation: :return
        requires :role, :foreground, coverage: :all_bars
        requires :role, :counterline
        requires :role, :color_anchor
      end
      # CONTINUATION_COLOR
      # CANDIDATE_CONTENT
    end
  end

  relation :depends_on,
           from: ref(:span, :continuation),
           to: ref(:span, :statement)
  relation :returns_to,
           from: ref(:span, :continuation),
           to: ref(:material, :theme)
end
