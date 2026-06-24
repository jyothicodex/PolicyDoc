/**
 * API service layer for PolicyAI
 * All authenticated requests include the JWT token from localStorage.
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const TOKEN_KEY = 'policyai_token';

/**
 * Get the stored JWT token.
 */
function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Helper for API fetch with authentication and error handling.
 */
export async function apiFetch(url, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      errorMsg = body.error || body.message || errorMsg;
    } catch {
      errorMsg = await response.text() || errorMsg;
    }
    throw new Error(errorMsg);
  }

  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

// ─────────────────────────────────────────────
// AUTH
// ─────────────────────────────────────────────

/**
 * Login with email and password.
 * Returns { token, requiresTwoFa, name, email, role, ... }
 */
export async function loginWithCredentials(email, password) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Login failed');
  }
  return data;
}

/**
 * Verify TOTP code during login (after receiving requiresTwoFa=true).
 * Requires the pre-auth token in Authorization header.
 */
export async function verifyTwoFaLogin(preAuthToken, code) {
  const response = await fetch(`${API_BASE}/auth/2fa/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${preAuthToken}`,
    },
    body: JSON.stringify({ code }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || '2FA verification failed');
  }
  return data;
}

/**
 * Change the current user's password.
 */
export async function changePassword(currentPassword, newPassword) {
  return apiFetch('/auth/change-password', {
    method: 'POST',
    body: JSON.stringify({ currentPassword, newPassword }),
  });
}

/**
 * Initiate 2FA setup — returns { secret, qrCodeBase64, otpAuthUri }
 */
export async function setup2FA() {
  return apiFetch('/auth/2fa/setup', { method: 'POST' });
}

/**
 * Confirm 2FA setup with a TOTP code to activate it.
 */
export async function verifySetup2FA(code) {
  return apiFetch('/auth/2fa/verify-setup', {
    method: 'POST',
    body: JSON.stringify({ code }),
  });
}

/**
 * Disable 2FA (requires password confirmation).
 */
export async function disable2FA(password) {
  return apiFetch('/auth/2fa/disable', {
    method: 'POST',
    body: JSON.stringify({ password }),
  });
}

/**
 * Delete the user's account permanently.
 */
export async function deleteAccount(password) {
  return apiFetch('/auth/account', {
    method: 'DELETE',
    body: JSON.stringify({ password }),
  });
}

// ─────────────────────────────────────────────
// DOCUMENTS
// ─────────────────────────────────────────────

export async function uploadDocument(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  let progressInterval = setInterval(() => {
    onProgress?.((prev) => Math.min((prev || 0) + 10, 90));
  }, 200);

  const token = getToken();
  try {
    const response = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      headers: token ? { 'Authorization': `Bearer ${token}` } : {},
      body: formData,
    });

    clearInterval(progressInterval);
    onProgress?.(100);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Upload failed');
    }

    return response.json();
  } catch (error) {
    clearInterval(progressInterval);
    throw error;
  }
}

export async function getDocuments() {
  return apiFetch('/documents');
}

export async function getDocument(id) {
  return apiFetch(`/documents/${id}`);
}

export async function getDocumentSummary(id) {
  return apiFetch(`/documents/${id}/summary`);
}

export async function deleteDocument(id) {
  await apiFetch(`/documents/${id}`, { method: 'DELETE' });
  return true;
}

export async function searchDocuments(query) {
  return apiFetch(`/documents/search?q=${encodeURIComponent(query)}`);
}

// ─────────────────────────────────────────────
// CHAT
// ─────────────────────────────────────────────

export async function askQuestion(question, documentId = null) {
  const body = { question };
  if (documentId && documentId !== 'global') body.documentId = documentId;
  return apiFetch('/chat/ask', { method: 'POST', body: JSON.stringify(body) });
}

export async function streamQuestion(question, documentId = null, onChunk, onMetadata, onDone, onError) {
  const body = { question };
  if (documentId && documentId !== 'global') body.documentId = documentId;
  
  const token = getToken();
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      let newlineIndex;
      while ((newlineIndex = buffer.indexOf('\n\n')) >= 0) {
        const eventText = buffer.slice(0, newlineIndex);
        buffer = buffer.slice(newlineIndex + 2);
        
        const lines = eventText.split('\n');
        let currentEventName = 'message';
        let dataLines = [];
        
        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEventName = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            dataLines.push(line.substring(5));
          }
        }
        
        if (dataLines.length > 0) {
          const data = dataLines.join('\n');
          if (currentEventName === 'metadata') {
            try { onMetadata(JSON.parse(data)); } catch(e) {}
          } else if (currentEventName === 'done') {
            onDone();
            return;
          } else {
            onChunk(data);
          }
        }
      }
    }
  } catch (e) {
    onError(e);
  }
}

export async function getChatHistory(documentId = 'global') {
  const url = documentId && documentId !== 'global'
    ? `/chat/history/${documentId}`
    : '/chat/history';
  return apiFetch(url);
}

export async function clearChatHistory(documentId = 'global') {
  const url = documentId && documentId !== 'global'
    ? `/chat/history/${documentId}`
    : '/chat/history';
  return apiFetch(url, { method: 'DELETE' });
}

export async function clearAllChatHistory() {
  return apiFetch('/chat/history', { method: 'DELETE' });
}

// ─────────────────────────────────────────────
// DASHBOARD & USERS
// ─────────────────────────────────────────────

export async function getDashboardStats() {
  return apiFetch('/dashboard/stats');
}

export async function getUserProfile() {
  return apiFetch('/users/me');
}

export async function updateUserProfile(name, email) {
  return apiFetch('/users/me', {
    method: 'PUT',
    body: JSON.stringify({ name, email }),
  });
}

// ─────────────────────────────────────────────
// NOTIFICATIONS
// ─────────────────────────────────────────────

export async function getNotifications() {
  return apiFetch('/notifications');
}

export async function markNotificationsAsRead() {
  return apiFetch('/notifications/read-all', { method: 'POST' });
}

// ─────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────

export async function exportDocumentsCSV() {
  const docs = await getDocuments();
  const headers = ['Name', 'Category', 'Pages', 'File Size (bytes)', 'Status', 'Uploaded At'];
  const rows = docs.map(d => [d.name, d.category, d.pageCount, d.fileSize, d.status, d.uploadedAt]);
  const csv = [headers, ...rows].map(r => r.map(v => `"${v}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'policyai-documents.csv';
  a.click();
  URL.revokeObjectURL(url);
}
