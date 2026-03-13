import { test, expect } from '@playwright/test';

test('has title', async ({ page }) => {
  await page.goto('/');

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/Banana Ripe Checker/);
});

test('check main text', async ({ page }) => {
  await page.goto('/');

  // Header의 BANANA와 Ripe Checker 텍스트가 보이는지 확인
  await expect(page.getByText('BANANA')).toBeVisible();
  await expect(page.getByText('Ripe Checker')).toBeVisible();
});
