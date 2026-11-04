// Cloudflare Pages advanced-mode worker.
// 1. 301-redirect the *.pages.dev alias and the www host to the canonical
//    custom domain, preserving the path/query, so search engines see one host.
// 2. Mark every other non-canonical host (branch/preview deployments) noindex,
//    so previews never end up in search results.
// brisc.run itself is served normally from static assets.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname === "brisc-docs.pages.dev" ||
        url.hostname === "www.brisc.run") {
      url.hostname = "brisc.run";
      return Response.redirect(url.toString(), 301);
    }
    const response = await env.ASSETS.fetch(request);
    if (url.hostname === "brisc.run") {
      return response;
    }
    const headers = new Headers(response.headers);
    headers.set("X-Robots-Tag", "noindex");
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
