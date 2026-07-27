production_piece "Composition kernel study", id: :composition_kernel_study do
  meter "4/4"
  key "C"

  material :theme_a do
    identity pitch: "rising third answered by descending step",
             rhythm: "two half-note impulses"
  end

  roster do
    part :flute, "Flute", music21: "Flute", family: :woodwind
    part :cello, "Violoncello", music21: "Violoncello", family: :string
  end

  section :whole, "Statement and return", bars: 1..4, type: :binary do
    span :statement, bars: 1..2, texture: :melody_over_bass do
      plan do
        requires :material, :theme_a, relation: :statement
        requires :harmony, coverage: :all_bars
        requires :role, :foreground, coverage: :all_bars
        requires :role, :bass_line, coverage: :all_bars
      end

      chords "b1:C b2:G7"

      phrase :theme_statement, surface: :absolute,
              material: :theme_a, relation: :statement do
        pitch_bars "C5 E5 | G5 E5"
        rhythm_bars "2 2 | 2 2"
      end
      placement :theme_statement, id: :theme_statement_flute,
                part: :flute, role: :foreground, at: "bar 1 beat 1"

      phrase :statement_bass, surface: :absolute do
        pitch_bars "C3 G2"
        rhythm_bars "2 2"
      end
      placement :statement_bass, id: :statement_bass_cello,
                part: :cello, role: :bass_line, at: "bar 1 beat 1"
    end

    span :return, bars: 3..4, texture: :varied_return do
      plan do
        requires :material, :theme_a, relation: :return
        requires :role, :foreground, coverage: :all_bars
      end
    end
  end

  relation :depends_on,
           from: ref(:span, :return),
           to: ref(:span, :statement)
  relation :returns_to,
           from: ref(:span, :return),
           to: ref(:material, :theme_a)
end
