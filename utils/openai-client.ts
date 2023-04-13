import { OpenAI } from 'langchain/llms';
import { Configuration } from 'openai';

if (!process.env.OPENAI_API_KEY) {
  throw new Error('Missing OpenAI Credentials');
}

export const openai = new OpenAI({
  temperature: 0
}, new Configuration({
  basePath:process.env.OPENAI_API_BASE_URL,
  apiKey:process.env.OPENAI_API_KEY
}));
