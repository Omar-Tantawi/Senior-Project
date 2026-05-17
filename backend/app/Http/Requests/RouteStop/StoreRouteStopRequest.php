<?php

namespace App\Http\Requests\RouteStop;

use Illuminate\Foundation\Http\FormRequest;

class StoreRouteStopRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'route_id'  => 'required|integer|exists:route,route_id',
            'name'      => 'required|string|max:100',
            'stoporder' => 'required|integer|min:1',
        ];
    }
}
