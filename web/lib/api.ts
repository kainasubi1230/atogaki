export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type ApiResult = { status: number; body: unknown };

export async function postJSON(
  path: string,
  body: unknown,
  token?: string,
): Promise<ApiResult> {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    });
    return { status: response.status, body: await response.json() };
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown_error";
    return {
      status: 0,
      body: {
        detail: `network_error: ${message}`,
      },
    };
  }
}
