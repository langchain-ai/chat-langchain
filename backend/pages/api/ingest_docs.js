import weaviate from 'weaviate-client';
import fs from 'fs';
import path from 'path';
import { JSDOM } from 'jsdom';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Weaviate as WeaviateStore } from 'langchain_community/vectorstores';
import { OpenAIEmbeddings } from 'langchain_openai';
import { SQLRecordManager, index } from 'langchain/indexes';

export default async function handler(req, res) {
  try {
    const WEAVIATE_URL = process.env.WEAVIATE_URL;
    const WEAVIATE_API_KEY = process.env.WEAVIATE_API_KEY;
    const RECORD_MANAGER_DB_URL = process.env.RECORD_MANAGER_DB_URL;
    const MY_HTML_FILES_DIRECTORY = process.env.MY_HTML_FILES_DIRECTORY;
    const WEAVIATE_DOCS_INDEX_NAME = process.env.WEAVIATE_DOCS_INDEX_NAME;

    const textSplitter = new RecursiveCharacterTextSplitter({ chunk_size: 4000, chunk_overlap: 200 });
    const embedding = new OpenAIEmbeddings({ model: "text-embedding-ada-002", chunk_size: 200 });

    const client = weaviate.client({
      scheme: 'https',
      host: WEAVIATE_URL,
      apiKey: new weaviate.ApiKey(WEAVIATE_API_KEY),
    });

    const vectorstore = new WeaviateStore({
      client,
      indexName: WEAVIATE_DOCS_INDEX_NAME,
      textKey: 'text',
      embedding,
      byText: false,
      attributes: ['source', 'title'],
    });

    const recordManager = new SQLRecordManager({
      db_url: RECORD_MANAGER_DB_URL,
      tableName: `weaviate/${WEAVIATE_DOCS_INDEX_NAME}`,
    });
    await recordManager.createSchema();

    const documents = [];
    const files = fs.readdirSync(MY_HTML_FILES_DIRECTORY);

    for (const file of files) {
      if (file.endsWith('.html')) {
        const filePath = path.join(MY_HTML_FILES_DIRECTORY, file);
        const htmlContent = fs.readFileSync(filePath, 'utf-8');
        const dom = new JSDOM(htmlContent);
        const text = dom.window.document.body.textContent.trim();
        const metadata = {
          source: filePath,
          title: dom.window.document.title,
          description: dom.window.document.querySelector('meta[name="description"]')?.content || '',
          language: dom.window.document.documentElement.lang || '',
        };
        documents.push({ page_content: text, metadata });
      }
    }

    const docsTransformed = textSplitter.splitDocuments(documents).filter(doc => doc.page_content.length > 10);

    for (const doc of docsTransformed) {
      if (!doc.metadata.source) doc.metadata.source = '';
      if (!doc.metadata.title) doc.metadata.title = '';
    }

    const indexingStats = await index(docsTransformed, recordManager, vectorstore, {
      cleanup: 'full',
      source_id_key: 'source',
      force_update: (process.env.FORCE_UPDATE || 'false').toLowerCase() === 'true',
    });

    res.status(200).json({ message: 'Ingestion successful', stats: indexingStats });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
}
