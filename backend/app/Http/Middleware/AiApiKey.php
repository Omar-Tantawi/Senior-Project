<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;

class AiApiKey
{
    public function handle(Request $request, Closure $next)
    {
        $expected = config('services.ai_api_key');

        if (!$expected || $request->header('X-AI-Key') !== $expected) {
            return response()->json(['error' => 'Unauthorized'], 401);
        }

        return $next($request);
    }
}
