import { z } from "zod";
import { OpenAIEmbeddings } from "@langchain/openai";

export const groqChatCompletionSchema = z.object({
  id: z.string(),
  object: z.string(),
  created: z.number(),
  model: z.string(),
  choices: z.array(
    z.object({
      index: z.number(),
      message: z.object({
        role: z.string(),
        content: z.string(),
      }),
      logprobs: z.null(),
      finish_reason: z.string(),
    })
  ),
  usage: z.object({
    queue_time: z.number(),
    prompt_tokens: z.number(),
    prompt_time: z.number(),
    completion_tokens: z.number(),
    completion_time: z.number(),
    total_tokens: z.number(),
    total_time: z.number(),
  }),
  system_fingerprint: z.string(),
  x_groq: z.object({
    id: z.string(),
  }),
});

export const groqEmbeddingSchema = z.object({
  object: z.string(),
  data: z.array(
    z.object({
      object: z.string(),
      embedding: z.array(z.number()),
      index: z.number(),
    })
  ),
  model: z.string(),
  usage: z.object({
    prompt_tokens: z.number(),
    total_tokens: z.number(),
  }),
});

export type GroqChatCompletion = z.infer<typeof groqChatCompletionSchema>;
export type GroqEmbedding = z.infer<typeof groqEmbeddingSchema>;

/**
 * Creates embeddings for text input using OpenAI's embedding API
 */
export async function createEmbedding(text: string) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OpenAI API key is not configured');
  }

  if (!process.env.EMBEDDING_MODEL) {
    throw new Error('Embedding model is not configured');
  }

  if (!process.env.EMBEDDING_MODEL_DIMENSIONS) {
    throw new Error('Embedding model dimensions is not configured');
  }

  const model = process.env.EMBEDDING_MODEL;
  const dimensions = +process.env.EMBEDDING_MODEL_DIMENSIONS;

  if (!text || text.trim() === '') {
    throw new Error('Empty text cannot be embedded');
  }

  try {
    // Initialize OpenAI Embeddings
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: model,
      dimensions: dimensions,
    }, {
      basePath: process.env.OPENAI_API_BASE_URL,
    });

    // Get the embedding for the text
    const embeddingResult = await embeddings.embedQuery(text);
    
    return embeddingResult;
  } catch (error) {
    console.error('Embedding error:', error);
    throw error instanceof Error ? error : new Error(String(error));
  }
}

/**
 * Creates embeddings for multiple text inputs
 */
export async function createEmbeddings(texts: Record<string, string>) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OpenAI API key is not configured');
  }

  if (!process.env.EMBEDDING_MODEL) {
    throw new Error('Embedding model is not configured');
  }

  if (!process.env.EMBEDDING_MODEL_DIMENSIONS) {
    throw new Error('Embedding model dimensions is not configured');
  }

  const model = process.env.EMBEDDING_MODEL;
  const dimensions = +process.env.EMBEDDING_MODEL_DIMENSIONS;
  
  // Initialize OpenAI Embeddings
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: model,
    dimensions: dimensions,
  }, {
    basePath: process.env.OPENAI_API_BASE_URL,
  });
  
  const embeddingsResult: Record<string, number[]> = {};

  // Process each text input
  for (const [key, text] of Object.entries(texts)) {
    if (!text || text.trim() === '') continue;
    
    try {
      embeddingsResult[key] = await embeddings.embedQuery(text);
    } catch (error) {
      console.error(`Error creating embedding for ${key}:`, error);
    }
  }
  
  return embeddingsResult;
}

/**
 * Creates embeddings for multiple text inputs in a single API call
 */
export async function createEmbeddingsBatch(texts: string[]): Promise<number[][]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OpenAI API key is not configured');
  }

  if (!texts.length) {
    return [];
  }
  
  // Filter out empty texts
  const validTexts = texts.filter(text => text && text.trim() !== '');
  
  if (!validTexts.length) {
    return [];
  }

  if (!process.env.EMBEDDING_MODEL) {
    throw new Error('Embedding model is not configured');
  }

  if (!process.env.EMBEDDING_MODEL_DIMENSIONS) {
    throw new Error('Embedding model dimensions is not configured');
  }

  const model = process.env.EMBEDDING_MODEL;
  const dimensions = +process.env.EMBEDDING_MODEL_DIMENSIONS;
  
  try {
    // Initialize OpenAI Embeddings
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: model,
      dimensions: dimensions,
    }, {
      basePath: process.env.OPENAI_API_BASE_URL,
    });

    // Get embeddings for all texts in a single batch request
    const embeddingResults = await embeddings.embedDocuments(validTexts);
    
    return embeddingResults;
  } catch (error) {
    console.error('Batch embedding error:', error);
    throw error instanceof Error ? error : new Error(String(error));
  }
}

// Define model type to allow any string (developer's choice)
type CompletionsParams = {
  model: string,
  prompt?: string,
  systemPrompt?: string,
  temperature: number,
  max_tokens?: number,
  embeddings?: Record<string, number[]>,
  userName?: string
};

// Define the request body type with proper TypeScript interface
interface GroqRequestBody {
  model: string;
  messages: Array<{role: string; content: string}>;
  temperature: number;
  max_tokens?: number;
  // Add any other standard properties here
  [key: string]: any; // Allow additional properties
}

/**
 * Generates text completions using Groq API
 */
export async function generateCompletions({ 
  model, 
  prompt, 
  systemPrompt, 
  temperature, 
  embeddings,
  userName
}: CompletionsParams) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OpenAI API Key is missing');
  }

  if (!process.env.OPENAI_API_BASE_URL) {
    throw new Error('OpenAI API Base URL is missing');
  }
  
  // If we have embeddings, incorporate them into the system prompt
  let enhancedSystemPrompt = systemPrompt || '';
  
  if (embeddings && Object.keys(embeddings).length > 0) {
    // Create a context section from the embeddings
    const contextSection = Object.entries(embeddings)
      .map(([key, value]) => {
        // We don't use the actual vector values, just the content they represent
        return `- ${key}`;
      })
      .join('\n');
    
    // Add the context to the system prompt
    if (enhancedSystemPrompt) {
      enhancedSystemPrompt = `${enhancedSystemPrompt}\n\nAdditional context for your reference:\n${contextSection}`;
    } else {
      enhancedSystemPrompt = `Use the following context to inform your response:\n${contextSection}`;
    }
  }
  
  // Enhance the system prompt for email generation to improve formatting
  if (enhancedSystemPrompt.toLowerCase().includes('email')) {
    enhancedSystemPrompt = process.env.AI_SYSTEM_PROMPT || `You are an email assistant helping ${userName} write professional and concise email replies.
  
  Important instructions:
  - Generate a real, ready-to-send email reply, not a template
  - Do not include placeholders like [Recipient], [discount percentage], etc.
  - Do not include formatting instructions or explanations
  - Do not include "Subject:" lines
  - Do not include "Here's a draft..." or similar meta-text
  - Write as if this email is ready to be sent immediately
  - Use real, specific content instead of placeholders
  - Address the recipient directly without using [brackets]
  - Be concise but thorough (2-3 paragraphs maximum)
  - Write in the first person as if you are ${userName}
  - Double space paragraphs (2 newlines)
  - Add two spaces below the sign-off
  - End with the name: ${userName}`;
  }
  
  // Ensure we have valid messages
  const messages = [];
  
  if (enhancedSystemPrompt) {
    messages.push({
      role: 'system',
      content: enhancedSystemPrompt,
    });
  }

  if (prompt && prompt.trim() !== '') {
    messages.push({ 
      role: 'user', 
      content: prompt 
    });
  } else {
    // If no prompt is provided, add a minimal prompt to avoid API errors
    messages.push({
      role: 'user',
      content: 'Please respond to this request with proper paragraph formatting.'
    });
  }

  // Prepare the request body with the correct type
  const requestBody: GroqRequestBody = {
    model: model,
    messages,
    temperature,
  };

  let url = process.env.OPENAI_API_BASE_URL;

  if (!url.endsWith('/')) {
    url += '/';
  }

  url += 'chat/completions';

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    
    try {
      const validatedData = groqChatCompletionSchema.parse(data);
      
      // Get the content
      let content = validatedData.choices[0]?.message.content || '';

      return { completion: content };
    } catch (validationError) {
      // Fall back to using the raw response if validation fails
      let content = data.choices[0]?.message.content || '';
      
      return { completion: content };
    }
  } catch (error) {
    console.error('API Call Error:', error);
    throw error;
  }
}

/**
 * Helper function to truncate email thread content to fit within token limits
 */
export function truncateThreadContent(threadContent: string, maxTokens: number = 12000): string {
  const emails = threadContent.split('\n---\n');
  let truncatedContent = emails[emails.length - 1];

  for (let i = emails.length - 2; i >= 0; i--) {
    const newContent = `${emails[i]}\n---\n${truncatedContent}`;
    const estimatedTokens = newContent.length / 4;

    if (estimatedTokens > maxTokens) {
      break;
    }

    truncatedContent = newContent;
  }

  return truncatedContent ?? '';
}