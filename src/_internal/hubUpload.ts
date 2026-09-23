/**
 * EXTENSION POINT — owned by the training builder (module "training").
 *
 * Explicit publication of a local artifact folder to the Hugging Face Hub
 * (Python ``HfApi.create_repo`` + ``upload_folder``). ``PretrainedModule.pushToHub``
 * calls this function; the training builder implements it (commit API with
 * LFS for large files) and must keep this signature.
 */
import { NotImplementedError } from '../errors.js';

export interface UploadFolderOptions {
  repoId: string;
  folderPath: string;
  private?: boolean;
  revision?: string | null;
  token?: string | null;
  commitMessage?: string;
  endpoint?: string | null;
  fetch?: typeof fetch;
}

export interface UploadResult {
  /** Commit URL or identifier reported by the Hub. */
  commit: string;
}

/** Create the model repository if needed and upload every file in ``folderPath``. */
export async function uploadFolder(options: UploadFolderOptions): Promise<UploadResult> {
  void options;
  throw new NotImplementedError('Hub publication is provided by the training module');
}
