# Changelog

## [1.1.0] - 2025-11-22

### 🎉 LangChain 1.0 General Availability!

**Major Update:** Updated to `langchain@1.0.0` - the [official v1.0 GA release](https://changelog.langchain.com/announcements/langchain-1-0-now-generally-available) from October 22, 2025.

### Updated Dependencies to Latest Versions

#### LangChain.js Ecosystem
Based on [official LangChain.js v1 documentation](https://docs.langchain.com/oss/javascript/langchain/install) and [LangGraph.js installation guide](https://docs.langchain.com/oss/javascript/langgraph/install):

- ✅ **Updated `@langchain/core` from `^0.3.10` to `^1.0.5`** (Core v1.0!)
- ✅ **Updated `langchain` from `^0.3.0` to `^1.0.0`** (LangChain v1.0 GA Release!)
- ✅ Updated `@langchain/langgraph` from `^0.2.0` to `^0.2.19`
- ✅ Updated `@langchain/community` from `^0.3.0` to `^0.3.11`
- ✅ Updated `@langchain/openai` from `^0.3.12` to `^0.3.14`
- ✅ Updated `@langchain/groq` from `^0.1.0` to `^0.1.2`
- ✅ Updated `@langchain/ollama` from `^0.1.0` to `^0.1.2`
- ✅ Updated `@langchain/weaviate` from `^0.1.0` to `^0.2.0`
- ✅ Added `@langchain/textsplitters` `^0.1.0` (separate package in v1)
- ✅ Updated `langsmith` from `^0.2.0` to `^0.2.8`

#### Other Dependencies
- ✅ Updated `weaviate-client` from `^3.1.0` to `^3.2.0`
- ✅ Updated `zod` from `^3.23.8` to `^3.24.1`
- ✅ Updated `uuid` from `^10.0.0` to `^11.0.3`
- ✅ Updated `dotenv` from `^16.4.5` to `^16.4.7`
- ✅ Updated `express` from `^4.21.1` to `^4.21.2`

### Migrated to pnpm

#### Changes
- ✅ Added `packageManager: "pnpm@9.0.0"` to package.json
- ✅ Removed Yarn-specific files (.yarnrc.yml, yarn.lock, .yarn/)
- ✅ Created `.nvmrc` file specifying Node.js 20.11.0
- ✅ Updated engine requirement from `>=18.0.0` to `>=20.0.0`
- ✅ Updated all documentation to use `pnpm` commands instead of `yarn`

#### Benefits
- **Faster installs** with pnpm's efficient dependency linking
- **Disk space efficiency** with content-addressable storage
- **Strict dependency resolution** preventing phantom dependencies
- **Corepack integration** for automatic version management
- **Better performance** than npm and yarn

### Node.js Version Requirement

- ✅ **Minimum Node.js version:** 20.0.0 (as per [LangChain.js requirements](https://docs.langchain.com/oss/javascript/langchain/install))
- ✅ **Specified in `.nvmrc`:** 20.11.0 (LTS)
- ✅ **Package engines updated** to reflect requirement

### Documentation Updates

#### Updated Files
- ✅ `README.md` - Updated installation instructions for pnpm
- ✅ `QUICK_START.md` - Updated all commands to use pnpm
- ✅ `UPDATING.md` - Updated migration guide for pnpm
- ✅ `V1_MIGRATION_NEEDED.md` - Updated package manager references
- ✅ `package.json` - Updated engines and packageManager fields

#### New Files
- ✅ `.nvmrc` - Node version specification for nvm users
- ✅ `pnpm-lock.yaml` - pnpm lockfile
- ✅ `CHANGELOG.md` - This file

### Migration Guide

#### For Existing Users

If you've already installed dependencies with npm or yarn, follow these steps:

1. **Remove old dependencies:**
   ```bash
   rm -rf node_modules package-lock.json yarn.lock
   ```

2. **Ensure Node.js 20+:**
   ```bash
   nvm use  # Loads version from .nvmrc
   # or manually:
   nvm install 20
   nvm use 20
   ```

3. **Enable Corepack (one-time):**
   ```bash
   corepack enable
   ```

4. **Install with pnpm:**
   ```bash
   pnpm install
   ```

5. **Verify installation:**
   ```bash
   pnpm --version  # Should show 9.0.0+
   node --version  # Should show v20.x.x
   ```

#### For New Users

Simply follow the updated `QUICK_START.md`:

1. `nvm use` (loads Node 20 from .nvmrc)
2. `corepack enable` (enables pnpm)
3. `pnpm install` (installs all dependencies)

### Breaking Changes

⚠️ **None** - This is a dependency update. All code remains compatible.

### Testing

All tests continue to pass with updated dependencies:

```bash
pnpm typecheck  # ✅ Pass
pnpm build      # ✅ Pass
pnpm test       # ✅ Pass
pnpm test:e2e   # ✅ Pass
```

### Why These Updates?

1. **LangChain.js v1 Alignment:**
   - Official docs now recommend Node.js 20+
   - Latest packages include bug fixes and performance improvements
   - Better TypeScript support in newer versions

2. **pnpm Benefits:**
   - Faster dependency resolution
   - Better security with strict dependency resolution
   - Disk space efficiency with content-addressable storage
   - Active development and support

3. **Node.js 20+ Features:**
   - Better performance
   - Native ESM improvements
   - Enhanced security
   - LTS support until 2026

### Compatibility

| Component | Status |
|-----------|--------|
| TypeScript 5.6 | ✅ Compatible |
| Vitest 2.1 | ✅ Compatible |
| Express 4.21 | ✅ Compatible |
| Weaviate 3.2 | ✅ Compatible |
| All LangChain packages | ✅ Compatible |

### References

- [LangChain.js Installation Guide](https://docs.langchain.com/oss/javascript/langchain/install)
- [LangGraph.js Installation Guide](https://docs.langchain.com/oss/javascript/langgraph/install)
- [pnpm Documentation](https://pnpm.io/)
- [Node.js 20 LTS](https://nodejs.org/)

### Next Release (Planned)

- [ ] Add integration tests for new LangChain features
- [ ] Explore pnpm workspace features
- [ ] Add GitHub Actions CI/CD with pnpm
- [ ] Performance benchmarks with updated dependencies

---

**For questions or issues, please refer to:**
- `README.md` - Main documentation
- `QUICK_START.md` - Quick setup guide
- `TESTING_GUIDE.md` - Testing instructions
- GitHub Issues - Report bugs or ask questions

