/**
 * Load an owned TensorCode chatbot from a local artifact or the Hugging Face
 * Hub and talk to it (Python ``examples/pretrained_chatbot.py``).
 *
 *     npm run build
 *     node examples/pretrainedChatbot.ts /tmp/chatbot-run/model --prompt 'Which evidence should we examine next?'
 *     node examples/pretrainedChatbot.ts jacob-valdez/tensorcode-chatbot-hotpot-001 --local-files-only
 *
 * Omit `--prompt` for an interactive session. `--load-session` / `--save-session`
 * restore and persist the conversation (session files are interchangeable with
 * Python's).
 */
import { createInterface } from 'node:readline/promises';
import { parseArgs } from 'node:util';
import { Chatbot } from 'tensorcode/tools';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    revision: { type: 'string' },
    device: { type: 'string', default: 'cpu' },
    'local-files-only': { type: 'boolean', default: false },
    prompt: { type: 'string' },
    'load-session': { type: 'string' },
    'save-session': { type: 'string' },
  },
});
const source = positionals[0];
if (!source) throw new Error('usage: pretrainedChatbot.ts MODEL [--prompt TEXT] [--revision REV] [--local-files-only]');

const model = await Chatbot.fromPretrained(source, {
  revision: values.revision ?? null, device: values.device, localFilesOnly: values['local-files-only'],
});
if (values['load-session']) await model.loadSession(values['load-session']);
if (values.prompt !== undefined) {
  console.log(model.call(values.prompt));
} else {
  const terminal = createInterface({ input: process.stdin, output: process.stdout });
  try {
    for (;;) {
      let prompt: string;
      try {
        prompt = await terminal.question('You: ');
      } catch {
        break; // end of input
      }
      if (prompt.trim()) console.log('Chatbot:', model.call(prompt));
    }
  } finally {
    terminal.close();
  }
}
if (values['save-session']) await model.saveSession(values['save-session']);
