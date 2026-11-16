const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load(env = {}, upstreamResponse) {
  const routes = {};
  const app = { use() {}, get(p, handler) { routes[p] = handler; }, post(p, handler) { routes[p] = handler; }, listen() {} };
  const express = () => app;
  express.json = () => () => {};
  let sent;
  let optionsSent;
  const axios = async (options) => {
    optionsSent = options;
    sent = options.data && JSON.parse(JSON.stringify(options.data));
    if (upstreamResponse) return upstreamResponse;
    return { status: 200, data: { choices: [{ message: {
      content: 'Answer', reasoning_content: 'Reasoning'
    } }] } };
  };
  axios.get = (url, options) => axios({ url, ...options });
  const context = vm.createContext({
    Buffer,
    require(name) {
      if (name === 'fs') return { existsSync: () => false };
      if (name === 'express') return express;
      if (name === 'cors') return () => () => {};
      if (name === 'axios') return axios;
      return require(name);
    },
    __dirname: path.join(__dirname, '..'),
    process: { env: { NIM_API_KEY: 'test', ...env } },
    console: { log() {}, warn() {}, error() {} }
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8'), context);
  return {
    context,
    async request(body, route = '/v1/chat/completions') {
      let result;
      let status;
      let output = '';
      let done;
      const ended = new Promise(resolve => { done = resolve; });
      await routes[route]({ body: {
        model: 'z-ai/glm-5.3', messages: [{ role: 'user', content: 'Hello' }], ...body
      } }, {
        status(value) { status = value; return this; },
        json(value) { result = value; },
        setHeader() {}, write(value) { output += value; }, end() { done(); }
      });
      if (body.stream && status === 200) await ended;
      return { sent, result, status, output, options: optionsSent };
    }
  };
}

test('GLM 5.3 sends raw defaults and displays reasoning', async () => {
  const { sent, result } = await load().request({});
  assert.deepEqual(sent.chat_template_kwargs, { clear_thinking: true });
  assert.equal(sent.reasoning_effort, 'low');
  assert.equal(sent.extra_body, undefined);
  assert.equal(sent.model, 'z-ai/glm-5.3');
  assert.equal(result.choices[0].message.content, '<think>\nReasoning\n</think>\n\nAnswer');
});

test('environment settings work, including capitalized False', async () => {
  const { sent } = await load({ CLEAR_THINKING: 'False', REASONING_EFFORT: 'high' }).request({});
  assert.equal(sent.chat_template_kwargs.clear_thinking, false);
  assert.equal(sent.reasoning_effort, 'high');
  assert.throws(() => load({ REASONING_EFFORT: 'none' }), /must be low, high, or max/);
});

test('request overrides survive flattening; top-level wins and model cannot change', async () => {
  const { sent } = await load().request({
    extra_body: { model: 'wrong', reasoning_effort: 'high', chat_template_kwargs: {
      clear_thinking: true, enable_thinking: false
    } },
    reasoning_effort: 'max', chat_template_kwargs: { clear_thinking: false }
  });
  assert.equal(sent.model, 'z-ai/glm-5.3');
  assert.equal(sent.reasoning_effort, 'max');
  assert.deepEqual(sent.chat_template_kwargs, { clear_thinking: false });
  const nested = await load().request({ extra_body: {
    reasoning_effort: 'high', chat_template_kwargs: { clear_thinking: false }
  } });
  assert.equal(nested.sent.reasoning_effort, 'high');
  assert.equal(nested.sent.chat_template_kwargs.clear_thinking, false);
});

test('other models retain their previous defaults', async () => {
  const { sent } = await load().request({ model: 'gpt-4o' });
  assert.equal(sent.model, 'qwen/qwen3-next-80b-a3b-thinking');
  assert.deepEqual(sent.extra_body, { chat_template_kwargs: { enable_thinking: true } });
  assert.equal(sent.reasoning_effort, undefined);
  assert.equal(sent.chat_template_kwargs, undefined);
});

test('streamed reasoning stays visible', () => {
  const { context } = load();
  const output = vm.runInContext(`(() => {
    const state = createStreamState();
    return reasoningChunkToContent({reasoning_content:'Reasoning'}, state, 0).content
      + reasoningChunkToContent({content:'Answer'}, state, 0).content;
  })()`, context);
  assert.equal(output, '<think>\nReasoning\n</think>\n\nAnswer');
});

const numesRoute = '/numes/v1/chat/completions';

test('Numes uses its own URL/key, keeps model IDs and does not inject NIM defaults', async () => {
  const { sent, options, status } = await load({ NUMES_API_KEY: 'numes-test' }).request({
    model: 'gpt-4o', reasoning: { effort: 'high' },
    extra_body: { model: 'wrong', temperature: 0.2 }, temperature: 0.5
  }, numesRoute);
  assert.equal(status, 200);
  assert.equal(options.url, 'https://api.numes.ai/api/v1/chat/completions');
  assert.equal(options.headers.Authorization, 'Bearer numes-test');
  assert.equal(sent.model, 'gpt-4o');
  assert.equal(sent.temperature, 0.5);
  assert.deepEqual(sent.reasoning, { effort: 'high' });
  assert.equal(sent.extra_body, undefined);
  assert.equal(sent.chat_template_kwargs, undefined);
  assert.equal(sent.reasoning_effort, undefined);
});

test('Numes requires configuration and an explicit model', async () => {
  assert.equal((await load().request({}, numesRoute)).status, 503);
  assert.equal((await load({ NUMES_API_KEY: 'test' }).request({ model: undefined }, numesRoute)).status, 400);
});

test('Numes model discovery uses its own provider', async () => {
  const response = { status: 200, data: { data: [{ id: 'provider/model' }] } };
  const { result, options } = await load({ NUMES_API_KEY: 'test' }, response).request({}, '/numes/v1/models');
  assert.deepEqual(result, response.data);
  assert.equal(options.url, 'https://api.numes.ai/api/v1/models');
  assert.equal(options.headers.Authorization, 'Bearer test');
});

test('structured reasoning can be preserved explicitly for capable clients', async () => {
  const details = [{ type: 'reasoning.encrypted', data: 'opaque' }];
  const response = { status: 200, data: { choices: [{ message: {
    content: 'Answer', reasoning: 'Thought', reasoning_details: details
  } }] } };
  for (const show of ['true', 'false']) {
    const { result } = await load({ NUMES_API_KEY: 'test', SHOW_REASONING: show, PRESERVE_REASONING_DETAILS: 'true' }, response).request({}, numesRoute);
    const message = result.choices[0].message;
    assert.equal(message.content, show === 'true' ? '<think>\nThought\n</think>\n\nAnswer' : 'Answer');
    assert.equal(message.reasoning, undefined);
    assert.deepEqual(JSON.parse(JSON.stringify(message.reasoning_details)), details);
  }
});

test('upstream errors remain JSON, including streaming HTML 403 responses', async () => {
  const { Readable } = require('node:stream');
  const response = { status: 403, data: Readable.from(['<html>Challenge</html>']) };
  const result = await load({ NUMES_API_KEY: 'test' }, response).request({ stream: true }, numesRoute);
  assert.equal(result.status, 403);
  assert.equal(result.result.error.type, 'upstream_error');
  const jsonError = { status: 429, data: { error: { message: 'Rate limited', code: 429 } } };
  assert.deepEqual((await load({ NUMES_API_KEY: 'test' }, jsonError).request({}, numesRoute)).result, jsonError.data);
});

test('Numes SSE handles split UTF-8 and closes reasoning on finish', async () => {
  const { Readable } = require('node:stream');
  const bytes = Buffer.from('data:{"choices":[{"index":0,"delta":{"reasoning":"考"}}]}\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  const split = bytes.indexOf(Buffer.from('考')) + 1;
  const response = { status: 200, headers: { 'content-type': 'text/event-stream' },
    data: Readable.from([bytes.subarray(0, split), bytes.subarray(split)]) };
  const { output } = await load({ NUMES_API_KEY: 'test' }, response).request({ stream: true }, numesRoute);
  const events = output.split('\n').filter(line => line.startsWith('data: ') && !line.includes('[DONE]')).map(line => JSON.parse(line.slice(6)));
  assert.equal(events.map(event => event.choices[0].delta.content).join(''), '<think>\n考\n</think>');
  assert.ok(output.endsWith('data: [DONE]\n\n'));
});


test('NIM model discovery retains its key and aliases', async () => {
  const response = { status: 200, data: { data: [{ id: 'nim/raw-model' }] } };
  const { result, options } = await load({}, response).request({}, '/v1/models');
  assert.equal(options.url, 'https://integrate.api.nvidia.com/v1/models');
  assert.equal(options.headers.Authorization, 'Bearer test');
  assert.ok(result.data.some(model => model.id === 'nim/raw-model'));
  assert.ok(result.data.some(model => model.id === 'gpt-4o'));
});


test('Janitor responses expose reasoning only once, including when hidden', async () => {
  const details = [{ type: 'reasoning.text', text: 'Thought' }];
  for (const show of ['true', 'false']) {
    const { context, request } = load({ SHOW_REASONING: show }, {
      status: 200, data: { choices: [{ message: {
        content: 'Answer', reasoning: 'Thought', reasoning_details: details
      } }] }
    });
    const { result } = await request({});
    assert.equal(result.choices[0].message.reasoning_details, undefined);
    const deltas = vm.runInContext(`(() => {
      const state = createStreamState();
      return [
        reasoningChunkToContent({ reasoning: 'The', reasoning_details: [{type:'reasoning.text',text:'The'}] }, state, 0),
        reasoningChunkToContent({ reasoning: ' thought', reasoning_details: [{type:'reasoning.text',text:' thought'}] }, state, 0),
        reasoningChunkToContent({ content: 'Answer' }, state, 0)
      ];
    })()`, context);
    for (const delta of deltas) {
      assert.equal(delta.reasoning, undefined);
      assert.equal(delta.reasoning_content, undefined);
      assert.equal(delta.reasoning_details, undefined);
    }
    assert.equal(deltas.map(d => d.content || '').join(''), show === 'true' ? '<think>\nThe thought\n</think>\n\nAnswer' : 'Answer');
  }
});
