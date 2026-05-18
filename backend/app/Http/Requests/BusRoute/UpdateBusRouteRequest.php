<?php

namespace App\Http\Requests\BusRoute;

use Illuminate\Foundation\Http\FormRequest;

class UpdateBusRouteRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'name' => 'sometimes|string|max:100|unique:route,name,' . $this->route('id') . ',route_id',
        ];
    }
}
