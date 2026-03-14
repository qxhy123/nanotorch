import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Latex } from '../components/ui/Latex';
import { TokenizationView } from '../components/visualization/tokenization/TokenizationView';
import { VocabularyBrowser } from '../components/visualization/tokenization/VocabularyBrowser';
import { TokenizerComparison } from '../components/visualization/tokenization/TokenizerComparison';
import { useTransformerStore } from '../stores/transformerStore';

function TokenizationMathReference() {
  const tokenizerType = useTransformerStore((state) => state.tokenizerType);
  const tokenizerVocabSize = useTransformerStore((state) => state.tokenizerVocabSize);
  const tokenizerNumMerges = useTransformerStore((state) => state.tokenizerNumMerges);
  const vocabularyData = useTransformerStore((state) => state.vocabularyData);
  const tokenizationResult = useTransformerStore((state) => state.tokenizationResult);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tokenization Reference</CardTitle>
        <CardDescription>
          Active tokenizer settings, current token count, and the text-to-id mapping pipeline.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Tokenizer</div>
            <div className="mt-2 text-2xl font-semibold">{tokenizerType.toUpperCase()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Target vocab</div>
            <div className="mt-2 text-2xl font-semibold">{tokenizerVocabSize.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">BPE merges</div>
            <div className="mt-2 text-2xl font-semibold">{tokenizerNumMerges.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Current tokens</div>
            <div className="mt-2 text-2xl font-semibold">{tokenizationResult?.tokens.length ?? 0}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Text to ids</div>
            <Latex display>{'x \\rightarrow (t_1, t_2, \\dots, t_n) \\rightarrow (id_1, id_2, \\dots, id_n)'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">BPE merge intuition</div>
            <Latex display>{'\\operatorname{merge}(a, b) \\rightarrow ab'}</Latex>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
          Loaded vocabulary size: {vocabularyData?.size ?? 0}. Special tokens and tokenizer-specific
          segmentation rules depend on the active tokenizer mode.
        </div>
      </CardContent>
    </Card>
  );
}

export default function TokenizationTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Tokenization learning path" />

      <div className="grid lg:grid-cols-2 gap-6">
        <TokenizationView />
        <DisclosureSection
          level="detailed"
          title="Vocabulary browser"
          description="Vocabulary inspection and token metadata unlock at the Detailed level."
        >
          <VocabularyBrowser />
        </DisclosureSection>
      </div>

      <DisclosureSection
        level="intermediate"
        title="Tokenizer comparison"
        description="Comparing char, word, and BPE behaviors unlocks at the Intermediate level."
      >
        <TokenizerComparison />
      </DisclosureSection>

      <DisclosureSection
        level="math"
        title="Tokenization reference"
        description="Text-to-id formulas and tokenizer settings unlock at the Math level."
      >
        <TokenizationMathReference />
      </DisclosureSection>
    </div>
  );
}
