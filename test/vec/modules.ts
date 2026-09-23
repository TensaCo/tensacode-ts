/** Small supplied modules shared by vector operation tests (Python test helpers). */
import { Module, Parameter, scalar, type Tensor } from '../../src/nn/index.js';

/** ``(query.unsqueeze(-2) * candidates).sum(-1) * scale`` with a trainable scale. */
export class DotScore extends Module {
  scale: Parameter;

  constructor(scale = 1) {
    super();
    this.scale = this.registerParameter('scale', new Parameter(scalar(scale)));
  }

  forward(query: Tensor, candidates: Tensor): Tensor {
    return query.unsqueeze(-2).mul(candidates).sum(-1).mul(this.scale);
  }
}

/** Parameter-free similarity. */
export class Similarity extends Module {
  forward(query: Tensor, candidates: Tensor): Tensor {
    return query.unsqueeze(-2).mul(candidates).sum(-1);
  }
}

/** A module whose behavior depends on a public JSON attribute. */
export class Scale extends Module {
  readonly factor: number;

  constructor(factor: number) {
    super();
    this.factor = factor;
  }

  forward(value: Tensor): Tensor {
    return value.mul(this.factor);
  }
}

/** Behavior declared through an explicit ``configuration()``. */
export class ExplicitScale extends Module {
  readonly factor: number;
  readonly opaqueRuntimeHelper = new Map<string, unknown>();

  constructor(factor: number) {
    super();
    this.factor = factor;
  }

  forward(value: Tensor): Tensor {
    return value.mul(this.factor);
  }

  configuration(): { factor: number } {
    return { factor: this.factor };
  }
}
