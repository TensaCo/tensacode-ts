/**
 * A trainable decision model using the shared investigation architecture
 * (Python ``tensorcode/tools/decision/__init__.py``).
 */
import { Investigator } from './investigator.js';

/**
 * Evaluate explicit candidates through an owned cognitive workspace.
 *
 * Decision shares the Investigator architecture and interface under a
 * distinct persisted tool identity. Construct from configuration or load
 * weights with ``fromPretrained``; application policy stays in caller code.
 */
export class Decision extends Investigator {
  static override readonly qualifiedName: string = 'tensorcode.tools.decision.Decision';
}
