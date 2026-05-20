import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const OUTPUT = path.resolve(process.cwd(), 'config.models.json');
const CDP_URL = process.env.WEBAI_CDP_URL || 'http://127.0.0.1:9222';
const STABLE_IDS = {
  lite: 'gemini-flash-lite-extended',
  flash: 'gemini-flash-extended',
  pro: 'gemini-pro-extended',
};

function classify(displayName) {
  const name = displayName.toLowerCase();
  if (name.includes('flash-lite')) return 'lite';
  if (name.includes('flash')) return 'flash';
  if (name.includes('pro')) return 'pro';
  return null;
}

function extractThinkingLevel(jslog) {
  if (!jslog) return null;
  const match = jslog.match(/\["[a-f0-9]+",(\d+),(\d+)\]/i);
  if (!match) return null;
  return { thinkingLevel: Number(match[1]), familyCode: Number(match[2]) };
}

async function ensureMenu(page) {
  if (await page.locator('gem-menu-item[data-mode-id]').count()) return;
  await page.locator('button.input-area-switch').last().click();
  await page.waitForTimeout(800);
}

async function readModels(page) {
  await ensureMenu(page);
  return await page.locator('gem-menu-item[data-mode-id]').evaluateAll(nodes => nodes.map(el => ({
    displayName: (el.querySelector('.label')?.textContent || el.textContent || '').trim().split('\n')[0].trim(),
    modeId: el.getAttribute('data-mode-id'),
  })));
}

async function readExtendedThinking(page, modeId) {
  await ensureMenu(page);
  await page.locator(`gem-menu-item[data-mode-id="${modeId}"]`).first().click();
  await page.waitForTimeout(500);
  await ensureMenu(page);
  const thinking = page.locator('gem-menu-item[value="thinking_level"]').first();
  await thinking.click();
  await page.waitForTimeout(500);
  const rows = await page.locator('gem-menu-item').evaluateAll(nodes => nodes.map(el => ({
    text: (el.innerText || el.textContent || '').trim(),
    jslog: el.getAttribute('jslog'),
  })));
  const extended = rows.find(row => row.text.includes('扩展') || /extended/i.test(row.text));
  return extractThinkingLevel(extended?.jslog) || { thinkingLevel: 2, familyCode: null };
}

async function main() {
  const browser = await chromium.connectOverCDP(CDP_URL);
  const context = browser.contexts()[0] || await browser.newContext();
  let page = context.pages().find(p => p.url().includes('gemini.google.com'));
  if (!page) page = await context.newPage();
  await page.goto('https://gemini.google.com/app', { waitUntil: 'domcontentloaded' }).catch(() => {});
  await page.bringToFront();
  await page.waitForTimeout(2000);

  const discovered = [];
  for (const model of await readModels(page)) {
    const kind = classify(model.displayName);
    if (!kind) continue;
    const thinking = await readExtendedThinking(page, model.modeId);
    discovered.push({
      id: STABLE_IDS[kind],
      displayName: model.displayName,
      modeId: model.modeId,
      thinkingLevel: thinking.thinkingLevel,
      familyCode: thinking.familyCode,
    });
  }

  const order = [STABLE_IDS.flash, STABLE_IDS.pro, STABLE_IDS.lite];
  discovered.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));

  if (discovered.length !== 3) {
    throw new Error(`Expected 3 Gemini Web models, discovered ${discovered.length}: ${JSON.stringify(discovered)}`);
  }

  const payload = { models: discovered, updatedAt: new Date().toISOString(), source: 'gemini-web-cdp' };
  fs.writeFileSync(OUTPUT, JSON.stringify(payload, null, 2) + '\n', 'utf8');
  console.log(`Wrote ${OUTPUT}`);
  console.log(JSON.stringify(payload, null, 2));
  await browser.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
