import type { Tutorial } from '../types/transformer';

/**
 * Transformer Basics Tutorial
 *
 * A comprehensive tutorial introducing the Transformer architecture.
 */
export const transformerBasicsTutorial: Tutorial = {
  id: 'transformer-basics',
  title: 'Transformer Architecture Basics',
  description: 'Learn the fundamentals of the Transformer architecture and how it processes sequences.',
  targetAudience: 'beginner',
  estimatedTime: 10,
  prerequisites: ['Basic understanding of neural networks'],
  steps: [
    {
      id: 'welcome',
      title: 'Welcome to the Transformer Visualizer!',
      content: `
        This interactive tutorial will guide you through the Transformer architecture,
        explaining how it processes and transforms text data.

        The Transformer is a neural network architecture that revolutionized natural language processing.
        It uses a mechanism called "attention" to understand relationships between words in a sequence.
      `,
      position: 'center',
    },
    {
      id: 'input-text',
      title: 'Input Text',
      content: `
        Start by entering some text in the input panel. This will be the data that the Transformer processes.

        Try entering a simple sentence like "The cat sat on the mat."
      `,
      target: '#input-panel',
      position: 'bottom',
      action: {
        type: 'input',
        selector: '#input-panel textarea',
      },
    },
    {
      id: 'embedding',
      title: 'Token Embeddings',
      content: `
        The first step is converting each word (token) into a numerical vector called an embedding.
        These embeddings capture the semantic meaning of each word.

        Words with similar meanings will have similar embeddings.
      `,
      target: '#embedding-visualization',
      position: 'left',
      highlightElements: ['#embedding-visualization', '.token-heatmap'],
    },
    {
      id: 'positional-encoding',
      title: 'Positional Encoding',
      content: `
        Since the Transformer processes all words simultaneously (not in sequence),
        we add positional encodings to tell the model where each word appears in the sentence.

        This uses sine and cosine functions at different frequencies.
      `,
      target: '#positional-encoding-visualization',
      position: 'right',
      highlightElements: ['#positional-encoding-visualization', '.positional-heatmap'],
    },
    {
      id: 'attention-intro',
      title: 'Self-Attention Mechanism',
      content: `
        The heart of the Transformer is the self-attention mechanism. It allows each word
        to "attend" to all other words in the sentence, figuring out which words are
        relevant to understanding it.

        For example, in "The cat sat on the mat", the word "sat" should attend strongly
        to "cat" (what sat?) and "mat" (where?).
      `,
      target: '#attention-visualization',
      position: 'bottom',
    },
    {
      id: 'multi-head-attention',
      title: 'Multi-Head Attention',
      content: `
        The Transformer uses multiple "heads" of attention simultaneously. Each head
        can learn to focus on different types of relationships:

        • Some heads focus on syntax (grammar)
        • Some focus on semantic relationships
        • Some focus on coreference (linking pronouns to nouns)

        You can switch between heads using the selector at the top.
      `,
      target: '#multi-head-attention',
      position: 'top',
      action: {
        type: 'click',
        selector: '.head-selector',
      },
    },
    {
      id: 'attention-stages',
      title: 'Attention Computation Stages',
      content: `
        Click on "Staged View" to see how attention is computed step-by-step:

        1. **Query (Q)**: What each token is looking for
        2. **Key (K)**: What each token contains
        3. **Value (V)**: The information to be extracted
        4. **Attention**: Q·K^T → Scale → Softmax → Weighted Sum

        Use the Next/Previous buttons to step through the computation!
      `,
      target: '#staged-attention-view',
      position: 'left',
      action: {
        type: 'click',
        selector: '.view-mode-staged',
      },
    },
    {
      id: 'feed-forward',
      title: 'Feed-Forward Network',
      content: `
        After attention, each position passes through a feed-forward network.
        This processes the attended information and transforms it into a useful representation.

        Think of it as each token "thinking about" what it learned from attention.
      `,
      target: '#feed-forward-visualization',
      position: 'right',
    },
    {
      id: 'layer-stack',
      title: 'Stacked Layers',
      content: `
        The Transformer repeats the attention + feed-forward pattern multiple times
        (typically 6 layers). Each layer can build upon the representations from
        previous layers.

        Lower layers learn simple patterns, while higher layers learn more abstract relationships.
      `,
      target: '#layer-stack-visualization',
      position: 'bottom',
    },
    {
      id: 'data-flow',
      title: 'Data Flow Diagram',
      content: `
        The Sankey diagram shows how data flows through the entire Transformer.
        Each node represents a transformation, and the links show the connections.

        Hover over nodes and links to see details about shapes and parameters.
      `,
      target: '#sankey-diagram',
      position: 'top',
    },
    {
      id: 'disclosure-levels',
      title: 'Disclosure Levels',
      content: `
        Use the disclosure level selector to control how much detail is shown:

        • **Overview**: High-level components only
        • **Intermediate**: Adds formulas and details
        • **Detailed**: Shows implementation details
        • **Math**: Full mathematical derivations

        Start with Overview and increase detail as you learn!
      `,
      target: '#disclosure-level-selector',
      position: 'bottom',
    },
    {
      id: 'conclusion',
      title: 'Tutorial Complete!',
      content: `
        Congratulations! You've learned the basics of the Transformer architecture.

        **Key Takeaways:**
        • Words are converted to embeddings
        • Positional encoding adds order information
        • Self-attention lets words relate to each other
        • Multiple heads capture different relationship types
        • Layers are stacked to build complex understanding

        **Next Steps:**
        • Try different input texts
        • Explore the staged attention view
        • Experiment with model parameters
        • Check out the math explanations at higher disclosure levels
      `,
      position: 'center',
    },
  ],
};

/**
 * Advanced Attention Tutorial
 */
export const advancedAttentionTutorial: Tutorial = {
  id: 'advanced-attention',
  title: 'Advanced Attention Mechanisms',
  description: 'Deep dive into attention mechanisms, including scaled dot-product, multi-head attention, and more.',
  targetAudience: 'advanced',
  estimatedTime: 15,
  prerequisites: ['transformer-basics'],
  steps: [
    {
      id: 'scaled-dot-product',
      title: 'Scaled Dot-Product Attention',
      content: `
        The core attention mechanism computes the compatibility between queries and keys:

        $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

        **Why scale by √dₖ?**
        For large values of dₖ, the dot products grow large, pushing softmax into
        regions with extremely small gradients. The scaling counteracts this.
      `,
      target: '#attention-formula',
      position: 'bottom',
    },
    {
      id: 'why-multi-head',
      title: 'Why Multiple Heads?',
      content: `
        Multi-head attention allows the model to attend to information from different
        representation subspaces simultaneously.

        Without multiple heads, each position would need to contain all information
        in a single attention mechanism, which could be limiting.
      `,
      target: '#multi-head-explanation',
      position: 'left',
    },
    {
      id: 'cross-attention',
      title: 'Cross-Attention in Enc-Dec',
      content: `
        In encoder-decoder Transformers, cross-attention allows the decoder to
        attend to the encoder's outputs.

        This is how the model "looks at" the input while generating each output token.
      `,
      target: '#cross-attention-viz',
      position: 'right',
    },
    {
      id: 'causal-masking',
      title: 'Causal Masking',
      content: `
        In decoders, we use causal masking to prevent attending to future tokens.
        This ensures autoregressive generation - each token can only see tokens
        before it.

        The mask sets future positions to -∞ before softmax, making their attention
        weights zero.
      `,
      target: '#mask-visualization',
      position: 'bottom',
    },
  ],
};

/**
 * Math Tutorial
 */
export const transformerMathTutorial: Tutorial = {
  id: 'transformer-math',
  title: 'Mathematical Foundations',
  description: 'Understand the mathematics behind the Transformer architecture.',
  targetAudience: 'advanced',
  estimatedTime: 20,
  prerequisites: ['transformer-basics', 'Basic linear algebra'],
  steps: [
    {
      id: 'embedding-math',
      title: 'Embedding Mathematics',
      content: `
        Token embedding is a lookup operation:

        $$E = XW_E$$

        Where X is the one-hot encoded token and W_E is the embedding matrix.
      `,
      target: '#embedding-math',
      position: 'bottom',
    },
    {
      id: 'positional-math',
      title: 'Positional Encoding Formula',
      content: `
        Positional encodings use sinusoidal functions:

        $$PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d_{model}})$$
        $$PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d_{model}})$$

        This choice allows the model to easily learn to attend by relative positions.
      `,
      target: '#positional-math',
      position: 'bottom',
    },
    {
      id: 'layer-norm-math',
      title: 'Layer Normalization',
      content: `
        Layer normalization stabilizes the hidden state dynamics:

        $$\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sigma} + \\beta$$

        Where μ and σ are computed per feature across the sequence dimension.
      `,
      target: '#layer-norm-math',
      position: 'right',
    },
  ],
};

// Export all tutorials
export const allTutorials: Tutorial[] = [
  transformerBasicsTutorial,
  advancedAttentionTutorial,
  transformerMathTutorial,
];
