<?php

namespace App\Http\Requests\RouteStop;

use Illuminate\Foundation\Http\FormRequest;

class UpdateRouteStopRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'name'      => 'sometimes|string|max:100',
            'stoporder' => 'sometimes|integer|min:1',
        ];
    }
}
