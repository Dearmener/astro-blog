'use strict';

const obsidian = require('obsidian');

const DEFAULT_SETTINGS = {
  lineHeight: 1.75,
  paragraphSpacing: 8,
  fontSize: 16,
  enableInEditMode: true,
  enableInReadMode: true,
  autoRemoveBlankLines: false,
  autoFormatOnPaste: false,
  formatOptions: {
    removeExtraBlankLines: true,
    normalizeHeaders: true,
    normalizeLists: true,
    normalizeBlockquotes: true,
    normalizeCodeBlocks: true,
    normalizeBold: true,
    normalizeItalic: true,
    normalizeHighlight: true,
    normalizeLinks: true,
    normalizeMath: true,
    removeTrailingSpaces: true,
    ensureNewlineAtEnd: true
  }
};

class NotionStyleSpacingPlugin extends obsidian.Plugin {
  async onload() {
    await this.loadSettings();
    this.addSettingTab(new NotionStyleSpacingSettingTab(this.app, this));
    this.applyStyles();
    
    this.addCommand({
      id: 'remove-blank-lines',
      name: 'Remove blank lines between paragraphs',
      editorCallback: (editor) => {
        this.removeBlankLines(editor);
      }
    });
    
    this.addCommand({
      id: 'format-document',
      name: 'Format entire document (Notion style)',
      editorCallback: (editor) => {
        this.formatDocument(editor);
      }
    });
    
    this.addCommand({
      id: 'format-selection',
      name: 'Format selection (Notion style)',
      editorCallback: (editor) => {
        this.formatSelection(editor);
      }
    });
    
    this.addCommand({
      id: 'toggle-auto-format',
      name: 'Toggle auto-format on paste',
      callback: async () => {
        this.settings.autoFormatOnPaste = !this.settings.autoFormatOnPaste;
        await this.saveSettings();
        new obsidian.Notice(`Auto-format on paste: ${this.settings.autoFormatOnPaste ? 'ON' : 'OFF'}`);
      }
    });
    
    this.registerEvent(
      this.app.workspace.on('editor-paste', (evt, editor) => {
        if (this.settings.autoFormatOnPaste) {
          setTimeout(() => {
            const selection = editor.getSelection();
            if (selection) {
              this.formatSelection(editor);
            }
          }, 50);
        } else if (this.settings.autoRemoveBlankLines) {
          setTimeout(() => this.removeBlankLines(editor), 100);
        }
      })
    );
    
    this.registerEvent(
      this.app.workspace.on('layout-change', () => {
        this.applyStyles();
      })
    );
  }

  onunload() {
    this.removeStyles();
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    if (!this.settings.formatOptions) {
      this.settings.formatOptions = DEFAULT_SETTINGS.formatOptions;
    }
  }

  async saveSettings() {
    await this.saveData(this.settings);
    this.applyStyles();
  }

  formatContent(content) {
    const opts = this.settings.formatOptions;
    let result = content;
    
    if (opts.removeTrailingSpaces) {
      result = result.replace(/[ \t]+$/gm, '');
    }
    
    if (opts.normalizeHeaders) {
      result = result.replace(/^(#{1,6})([^\s#])/gm, '$1 $2');
      result = result.replace(/^(#{1,6})\s+/gm, (match, hashes) => hashes + ' ');
    }
    
    if (opts.normalizeBold) {
      result = result.replace(/\*\*\s+/g, '**');
      result = result.replace(/\s+\*\*/g, '**');
      result = result.replace(/__\s+/g, '__');
      result = result.replace(/\s+__/g, '__');
      result = result.replace(/\*\*([^*]+)\*\*/g, (match, content) => {
        return '**' + content.trim() + '**';
      });
    }
    
    if (opts.normalizeItalic) {
      result = result.replace(/(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)/g, (match, content) => {
        return '*' + content.trim() + '*';
      });
      result = result.replace(/(?<!_)_(?!_)([^_]+)(?<!_)_(?!_)/g, (match, content) => {
        return '_' + content.trim() + '_';
      });
    }
    
    if (opts.normalizeHighlight) {
      result = result.replace(/==\s+/g, '==');
      result = result.replace(/\s+==/g, '==');
      result = result.replace(/==([^=]+)==/g, (match, content) => {
        return '==' + content.trim() + '==';
      });
    }
    
    if (opts.normalizeLists) {
      result = result.replace(/^(\s*)[-*+]\s+/gm, '$1- ');
      result = result.replace(/^(\s*)(\d+)[.)]\s+/gm, '$1$2. ');
      result = result.replace(/^(\s*)-\s{2,}/gm, '$1- ');
      result = result.replace(/^(\s*\d+\.)\s{2,}/gm, '$1 ');
    }
    
    if (opts.normalizeBlockquotes) {
      result = result.replace(/^>\s*/gm, '> ');
      result = result.replace(/^>([^\s>])/gm, '> $1');
    }
    
    if (opts.normalizeCodeBlocks) {
      result = result.replace(/```\s*\n\s*\n/g, '```\n');
      result = result.replace(/\n\s*\n```/g, '\n```');
      result = result.replace(/`\s+/g, '`');
      result = result.replace(/\s+`/g, '`');
    }
    
    if (opts.normalizeMath) {
      result = result.replace(/\$\s+/g, '$');
      result = result.replace(/\s+\$/g, '$');
      result = result.replace(/\$\$\s*\n\s*\n/g, '$$\n');
      result = result.replace(/\n\s*\n\s*\$\$/g, '\n$$');
    }
    
    if (opts.normalizeLinks) {
      result = result.replace(/\[\s+/g, '[');
      result = result.replace(/\s+\]/g, ']');
      result = result.replace(/\]\s+\(/g, '](');
      result = result.replace(/\(\s+/g, '(');
      result = result.replace(/\s+\)/g, ')');
    }
    
    if (opts.removeExtraBlankLines) {
      const codeBlocks = [];
      result = result.replace(/```[\s\S]*?```/g, (match) => {
        codeBlocks.push(match);
        return `___CODEBLOCK_${codeBlocks.length - 1}___`;
      });
      
      const lines = result.split('\n');
      const filtered = lines.filter((line, i) => {
        const trimmed = line.trim();
        if (trimmed === '---' || trimmed === '***' || trimmed === '___') return false;
        if (trimmed !== '') return true;
        if (i === 0) return false;
        const prevLine = lines[i - 1].trim();
        const needsBlankAfter = /^```/.test(prevLine);
        return needsBlankAfter;
      });
      result = filtered.join('\n');
      
      result = result.replace(/___CODEBLOCK_(\d+)___/g, (match, index) => {
        return codeBlocks[parseInt(index)];
      });
    }
    
    if (opts.ensureNewlineAtEnd) {
      result = result.replace(/\n*$/, '\n');
    }
    
    return result;
  }

  formatDocument(editor) {
    const content = editor.getValue();
    const cursor = editor.getCursor();
    const newContent = this.formatContent(content);
    
    if (content !== newContent) {
      editor.setValue(newContent);
      const lineCount = newContent.split('\n').length;
      editor.setCursor({ 
        line: Math.min(cursor.line, lineCount - 1), 
        ch: cursor.ch 
      });
      new obsidian.Notice('Document formatted');
    } else {
      new obsidian.Notice('Document already formatted');
    }
  }

  formatSelection(editor) {
    const selection = editor.getSelection();
    if (!selection) {
      this.formatDocument(editor);
      return;
    }
    
    const newSelection = this.formatContent(selection);
    if (selection !== newSelection) {
      editor.replaceSelection(newSelection);
      new obsidian.Notice('Selection formatted');
    }
  }

  removeBlankLines(editor) {
    const content = editor.getValue();
    const cursor = editor.getCursor();
    
    const codeBlocks = [];
    let temp = content.replace(/```[\s\S]*?```/g, (match) => {
      codeBlocks.push(match);
      return `___CODEBLOCK_${codeBlocks.length - 1}___`;
    });
    
    const lines = temp.split('\n');
    const filtered = lines.filter((line, i) => {
      const trimmed = line.trim();
      if (trimmed === '---' || trimmed === '***' || trimmed === '___') return false;
      if (trimmed !== '') return true;
      if (i === 0) return false;
      const prevLine = lines[i - 1].trim();
      const needsBlankAfter = /^```/.test(prevLine);
      return needsBlankAfter;
    });
    let newContent = filtered.join('\n');
    
    newContent = newContent.replace(/___CODEBLOCK_(\d+)___/g, (match, index) => {
      return codeBlocks[parseInt(index)];
    });
    
    if (content !== newContent) {
      editor.setValue(newContent);
      const lineCount = newContent.split('\n').length;
      editor.setCursor({ 
        line: Math.min(cursor.line, lineCount - 1), 
        ch: cursor.ch 
      });
      new obsidian.Notice('Blank lines removed');
    }
  }

  applyStyles() {
    this.removeStyles();
    
    const ps = this.settings.paragraphSpacing;
    const lh = this.settings.lineHeight;
    const fs = this.settings.fontSize;
    
    const css = `
      /* ===== Editor Mode (Source / Live Preview) ===== */
      ${this.settings.enableInEditMode ? `
      
      .markdown-source-view.mod-cm6 .cm-editor {
        font-size: ${fs}px !important;
      }
      
      .markdown-source-view.mod-cm6 .cm-scroller {
        line-height: ${lh} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
      }
      
      .markdown-source-view.mod-cm6 .cm-content {
        padding-top: 20px !important;
        padding-bottom: 50vh !important;
      }
      
      .markdown-source-view.mod-cm6 .cm-line {
        padding-top: ${ps}px !important;
        padding-bottom: ${ps}px !important;
        line-height: ${lh} !important;
      }
      
      .markdown-source-view.mod-cm6 .cm-line:first-child {
        padding-top: 0 !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-header {
        padding-top: ${ps * 2}px !important;
        padding-bottom: ${ps}px !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-header-1 {
        padding-top: ${ps * 3}px !important;
        padding-bottom: ${ps * 1.5}px !important;
        line-height: 1.3 !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-header-2 {
        padding-top: ${ps * 2.5}px !important;
        padding-bottom: ${ps * 1.25}px !important;
        line-height: 1.35 !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-header-3 {
        padding-top: ${ps * 2}px !important;
        padding-bottom: ${ps}px !important;
        line-height: 1.4 !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-list-line {
        padding-top: ${ps * 0.5}px !important;
        padding-bottom: ${ps * 0.5}px !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-codeblock {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        line-height: 1.5 !important;
      }
      
      .markdown-source-view.mod-cm6 .HyperMD-quote {
        padding-top: ${ps * 0.75}px !important;
        padding-bottom: ${ps * 0.75}px !important;
      }
      
      .markdown-source-view.mod-cm6 .cm-embed-block {
        padding-top: ${ps}px !important;
        padding-bottom: ${ps}px !important;
      }
      
      ` : ''}
      
      /* ===== Reading Mode ===== */
      ${this.settings.enableInReadMode ? `
      
      .markdown-preview-view,
      .markdown-rendered {
        font-size: ${fs}px !important;
        line-height: ${lh} !important;
      }
      
      .markdown-preview-view p,
      .markdown-rendered p {
        margin-top: 0 !important;
        margin-bottom: ${ps * 2}px !important;
        line-height: ${lh} !important;
      }
      
      .markdown-preview-view li,
      .markdown-rendered li {
        margin-bottom: ${ps * 0.5}px !important;
        line-height: ${lh} !important;
      }
      
      .markdown-preview-view h1 {
        margin-top: ${ps * 4}px !important;
        margin-bottom: ${ps * 2}px !important;
      }
      
      .markdown-preview-view h2 {
        margin-top: ${ps * 3.5}px !important;
        margin-bottom: ${ps * 1.5}px !important;
      }
      
      .markdown-preview-view h3 {
        margin-top: ${ps * 3}px !important;
        margin-bottom: ${ps}px !important;
      }
      
      .markdown-preview-view h4,
      .markdown-preview-view h5,
      .markdown-preview-view h6 {
        margin-top: ${ps * 2.5}px !important;
        margin-bottom: ${ps}px !important;
      }
      
      .markdown-preview-view blockquote {
        margin: ${ps * 2}px 0 !important;
        padding: ${ps}px ${ps * 2}px !important;
        line-height: ${lh} !important;
      }
      
      .markdown-preview-view pre {
        margin: ${ps * 2}px 0 !important;
      }
      
      .markdown-preview-view ul,
      .markdown-preview-view ol {
        margin-top: 0 !important;
        margin-bottom: ${ps * 2}px !important;
      }
      
      .markdown-preview-view hr {
        margin: ${ps * 3}px 0 !important;
      }
      
      ` : ''}
    `;
    
    const styleEl = document.createElement('style');
    styleEl.id = 'notion-style-spacing';
    styleEl.textContent = css;
    document.head.appendChild(styleEl);
  }

  removeStyles() {
    const existing = document.getElementById('notion-style-spacing');
    if (existing) {
      existing.remove();
    }
  }
}

class NotionStyleSpacingSettingTab extends obsidian.PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();
    
    containerEl.createEl('h2', { text: 'Notion Style Spacing' });
    
    // ===== Spacing Settings =====
    containerEl.createEl('h3', { text: 'Spacing' });
    
    new obsidian.Setting(containerEl)
      .setName('Font size')
      .setDesc('Base font size in pixels')
      .addSlider(slider => slider
        .setLimits(12, 24, 1)
        .setValue(this.plugin.settings.fontSize)
        .setDynamicTooltip()
        .onChange(async (value) => {
          this.plugin.settings.fontSize = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Line height')
      .setDesc('Line height multiplier')
      .addSlider(slider => slider
        .setLimits(1.2, 2.5, 0.05)
        .setValue(this.plugin.settings.lineHeight)
        .setDynamicTooltip()
        .onChange(async (value) => {
          this.plugin.settings.lineHeight = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Paragraph spacing')
      .setDesc('Vertical padding in pixels')
      .addSlider(slider => slider
        .setLimits(0, 20, 1)
        .setValue(this.plugin.settings.paragraphSpacing)
        .setDynamicTooltip()
        .onChange(async (value) => {
          this.plugin.settings.paragraphSpacing = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Enable in Edit mode')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.enableInEditMode)
        .onChange(async (value) => {
          this.plugin.settings.enableInEditMode = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Enable in Reading mode')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.enableInReadMode)
        .onChange(async (value) => {
          this.plugin.settings.enableInReadMode = value;
          await this.plugin.saveSettings();
        }));
    
    // ===== Auto Format Settings =====
    containerEl.createEl('h3', { text: 'Auto Format' });
    
    new obsidian.Setting(containerEl)
      .setName('Auto-format on paste')
      .setDesc('Automatically format pasted content')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.autoFormatOnPaste)
        .onChange(async (value) => {
          this.plugin.settings.autoFormatOnPaste = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Auto-remove blank lines')
      .setDesc('Remove extra blank lines on paste (when auto-format is off)')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.autoRemoveBlankLines)
        .onChange(async (value) => {
          this.plugin.settings.autoRemoveBlankLines = value;
          await this.plugin.saveSettings();
        }));
    
    // ===== Format Options =====
    containerEl.createEl('h3', { text: 'Format Options' });
    containerEl.createEl('p', { 
      text: 'Configure what gets formatted',
      cls: 'setting-item-description'
    });
    
    const opts = this.plugin.settings.formatOptions;
    
    new obsidian.Setting(containerEl)
      .setName('Remove extra blank lines')
      .setDesc('Compress multiple blank lines into one')
      .addToggle(toggle => toggle
        .setValue(opts.removeExtraBlankLines)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.removeExtraBlankLines = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize headers')
      .setDesc('Ensure space after # symbols')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeHeaders)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeHeaders = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize lists')
      .setDesc('Standardize list markers (- for unordered, 1. for ordered)')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeLists)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeLists = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize bold')
      .setDesc('Remove extra spaces in **bold** text')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeBold)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeBold = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize italic')
      .setDesc('Remove extra spaces in *italic* text')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeItalic)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeItalic = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize highlight')
      .setDesc('Remove extra spaces in ==highlight== text')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeHighlight)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeHighlight = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize code')
      .setDesc('Clean up code blocks and inline code')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeCodeBlocks)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeCodeBlocks = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize math')
      .setDesc('Clean up $inline$ and $$block$$ math')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeMath)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeMath = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize links')
      .setDesc('Remove extra spaces in [links](url)')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeLinks)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeLinks = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Normalize blockquotes')
      .setDesc('Ensure proper > formatting')
      .addToggle(toggle => toggle
        .setValue(opts.normalizeBlockquotes)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.normalizeBlockquotes = value;
          await this.plugin.saveSettings();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Remove trailing spaces')
      .setDesc('Remove spaces at end of lines')
      .addToggle(toggle => toggle
        .setValue(opts.removeTrailingSpaces)
        .onChange(async (value) => {
          this.plugin.settings.formatOptions.removeTrailingSpaces = value;
          await this.plugin.saveSettings();
        }));
    
    // ===== Actions =====
    containerEl.createEl('h3', { text: 'Actions' });
    
    new obsidian.Setting(containerEl)
      .setName('Format current document')
      .setDesc('Apply all format options to the entire document')
      .addButton(button => button
        .setButtonText('Format')
        .setCta()
        .onClick(() => {
          const activeView = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
          if (activeView) {
            this.plugin.formatDocument(activeView.editor);
          } else {
            new obsidian.Notice('No active editor');
          }
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Remove blank lines only')
      .setDesc('Just remove extra blank lines without other formatting')
      .addButton(button => button
        .setButtonText('Remove')
        .onClick(() => {
          const activeView = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
          if (activeView) {
            this.plugin.removeBlankLines(activeView.editor);
          } else {
            new obsidian.Notice('No active editor');
          }
        }));
    
    // ===== Presets =====
    containerEl.createEl('h3', { text: 'Presets' });
    
    new obsidian.Setting(containerEl)
      .setName('Notion Default')
      .setDesc('Line height: 1.6, Spacing: 6px')
      .addButton(button => button
        .setButtonText('Apply')
        .onClick(async () => {
          this.plugin.settings.lineHeight = 1.6;
          this.plugin.settings.paragraphSpacing = 6;
          this.plugin.settings.fontSize = 16;
          await this.plugin.saveSettings();
          this.display();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Comfortable')
      .setDesc('Line height: 1.75, Spacing: 8px')
      .addButton(button => button
        .setButtonText('Apply')
        .onClick(async () => {
          this.plugin.settings.lineHeight = 1.75;
          this.plugin.settings.paragraphSpacing = 8;
          this.plugin.settings.fontSize = 16;
          await this.plugin.saveSettings();
          this.display();
        }));
    
    new obsidian.Setting(containerEl)
      .setName('Compact')
      .setDesc('Line height: 1.5, Spacing: 4px')
      .addButton(button => button
        .setButtonText('Apply')
        .onClick(async () => {
          this.plugin.settings.lineHeight = 1.5;
          this.plugin.settings.paragraphSpacing = 4;
          this.plugin.settings.fontSize = 15;
          await this.plugin.saveSettings();
          this.display();
        }));
    
    // ===== Commands Info =====
    containerEl.createEl('h3', { text: 'Commands (⌘+P)' });
    const cmdList = containerEl.createEl('ul', { cls: 'setting-item-description' });
    cmdList.createEl('li', { text: 'Format entire document (Notion style)' });
    cmdList.createEl('li', { text: 'Format selection (Notion style)' });
    cmdList.createEl('li', { text: 'Remove blank lines between paragraphs' });
    cmdList.createEl('li', { text: 'Toggle auto-format on paste' });
  }
}

module.exports = NotionStyleSpacingPlugin;
